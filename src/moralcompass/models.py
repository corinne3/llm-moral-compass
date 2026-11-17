"""Model backends.

All backends share the same small interface:

- option_probs(conversations, options) -> array (n, k+1)
      probability of each option token, renormalized over the k options,
      plus a last column "mass" = how much of the model's total probability
      went to the options at all (a format-validity check).
- hidden_states(texts)          -> mean activations per layer (open models)
- steer(interventions)          -> context manager that adds vectors/neuron
                                   values during forward passes (open models)
- neuron_residual_effect(...)   -> what a neuron intervention adds to the
                                   residual stream (to compare methods fairly)
- generate(messages)            -> short greedy text (sanity check)
- nll(texts)                    -> mean negative log-likelihood per token

Backends:
- HFModel:   a real open-weights model run locally with Hugging Face
             transformers (CPU is fine for 0.5B-1.5B models).
- MockModel: a tiny synthetic stand-in, ONLY for tests of the pipeline.
- OpenAICompatibleModel: optional plug for a hosted model (no steering).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import urllib.request

import numpy as np

from .interventions import AddToNeurons, AddToResidual


# =========================================================================
# Real model: Hugging Face transformers
# =========================================================================
class HFModel:
    def __init__(self, model_id: str, dtype: str = "float32", batch_size: int = 8,
                 device: str = "cpu", max_length: int = 1024, threads: int | None = None,
                 quant: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.name = model_id
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"          # last position = last real token
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                       "float16": torch.float16}[dtype]
        kwargs = {}
        if quant == "4bit":
            # 4-bit weights (bitsandbytes, GPU only): a 7B model fits in ~5.5 GB.
            # Activations stay in float16, so probes and steering work as usual.
            from transformers import BitsAndBytesConfig
            kwargs = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16), "device_map": {"": 0}}
        try:   # recent transformers: "dtype"; older ones: "torch_dtype"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        if quant is None:
            self.model.to(device)
        self.model.eval()
        self.layers = _find_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self._down = [_find_down_proj(layer) for layer in self.layers]
        self.d_mlp = self._down[0].in_features
        self._active: list = []                 # interventions currently applied
        self._hooks: list = []

    # ---------------------------------------------------------------- text
    def format_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """Apply the model's own chat template (each model has its own)."""
        if getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            except Exception:
                # some templates (e.g. Gemma) refuse a "system" role:
                # merge the system text into the first user message
                merged = _merge_system(messages)
                return self.tok.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=add_generation_prompt)
        text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
        return text + ("\nAssistant:" if add_generation_prompt else "")

    def _option_token_ids(self, options):
        """Token ids that spell each option, with and without a leading space."""
        ids = []
        for opt in options:
            found = set()
            for variant in (opt, " " + opt):
                enc = self.tok.encode(variant, add_special_tokens=False)
                if len(enc) == 1:
                    found.add(enc[0])
            if not found:
                raise ValueError(f"Option {opt!r} is not a single token for {self.name}")
            ids.append(sorted(found))
        return ids

    # ------------------------------------------------------------- scoring
    def _last_logits(self, enc, keep: int):
        """Logits of the last `keep` positions only. Computing the logits of
        every position (vocabulary ~150k) wastes GPU/CPU memory."""
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                return self.model(**enc, **{kw: keep}).logits[:, -keep:, :].float()
            except TypeError:
                continue
        return self.model(**enc).logits[:, -keep:, :].float()

    def option_probs(self, conversations: list[list[dict]], options=("1", "2")) -> np.ndarray:
        torch = self.torch
        texts = [self.format_chat(c) for c in conversations]
        opt_ids = self._option_token_ids(options)
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=False).to(self.device)
            with torch.inference_mode():
                logits = self._last_logits(enc, 1)[:, -1, :]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            p = np.stack([probs[:, ids].sum(axis=1) for ids in opt_ids], axis=1)
            mass = p.sum(axis=1, keepdims=True)
            out.append(np.concatenate([p / np.clip(mass, 1e-12, None), mass], axis=1))
        return np.concatenate(out, axis=0)

    def continuation_logprobs(self, prompts: list[str], continuations: list[str]) -> np.ndarray:
        """log P(continuation | prompt) = sum of the log-probabilities of the
        continuation's tokens (chain rule). Prompts are already formatted."""
        torch = self.torch
        out = []
        for i in range(0, len(prompts), self.batch_size):
            P, C = prompts[i:i + self.batch_size], continuations[i:i + self.batch_size]
            full = [p + c for p, c in zip(P, C)]
            n_cont = [len(self.tok(f, add_special_tokens=False)["input_ids"])
                      - len(self.tok(p, add_special_tokens=False)["input_ids"])
                      for p, f in zip(P, full)]
            keep = max(n_cont) + 1
            enc = self.tok(full, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=False).to(self.device)
            with torch.inference_mode():
                logits = self._last_logits(enc, keep)
            logp = torch.log_softmax(logits[:, :-1, :], dim=-1)          # predicts the next token
            target = enc["input_ids"][:, -(keep - 1):]
            tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for j, n in enumerate(n_cont):
                out.append(float(tok_lp[j, -n:].sum()))
        return np.asarray(out)

    def cloze_logprobs(self, conversations: list[list[dict]], option_pairs: list[tuple],
                       prefix: str) -> np.ndarray:
        """For each conversation, log P(" <option>.") after the assistant
        starts its answer with `prefix`. Returns an array (n, 2)."""
        prompts, conts = [], []
        for conv, pair in zip(conversations, option_pairs):
            base = self.format_chat(conv) + prefix
            for opt in pair:
                prompts.append(base)
                conts.append(" " + opt + ".")
        return self.continuation_logprobs(prompts, conts).reshape(-1, 2)

    # --------------------------------------------------------- activations
    def hidden_states(self, texts: list[str], mlp: bool = True) -> dict:
        """Mean over tokens of (a) the residual stream after each decoder
        layer and (b) the MLP neuron activations of each layer."""
        torch = self.torch
        resid = {l: [] for l in range(self.n_layers)}
        neurons = {l: [] for l in range(self.n_layers)}
        norms = {l: [] for l in range(self.n_layers)}
        store: dict = {}

        def out_hook(l):
            def fn(module, inputs, output):
                store[("r", l)] = output[0] if isinstance(output, tuple) else output
            return fn

        def mlp_hook(l):
            def fn(module, inputs):
                store[("m", l)] = inputs[0]
            return fn

        handles = [self.layers[l].register_forward_hook(out_hook(l)) for l in range(self.n_layers)]
        if mlp:
            handles += [self._down[l].register_forward_pre_hook(mlp_hook(l))
                        for l in range(self.n_layers)]
        try:
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_length).to(self.device)
                am = enc["attention_mask"].clone()
                # Exclude the FIRST real token: it is an "attention sink" with a
                # massive activation (hundreds of times the others) that would
                # dominate any average (Sun et al. 2024, "Massive activations").
                first = am.argmax(dim=1)                      # left padding: first real token
                am[torch.arange(am.shape[0]), first] = 0
                with torch.inference_mode():
                    self.model(**enc)
                for l in range(self.n_layers):
                    h = store[("r", l)].float()
                    # also drop any other "massive" token of this layer (> 10x the median)
                    tok_norm = h.norm(dim=-1)                 # (batch, tokens)
                    med = tok_norm[am.bool()].median()
                    keep = am.bool() & (tok_norm < 10 * med)
                    mk = keep.unsqueeze(-1).float()
                    denom = mk.sum(1).clamp(min=1)
                    resid[l].append(((h * mk).sum(1) / denom).cpu().numpy())
                    norms[l].extend(tok_norm[keep].cpu().tolist())
                    if mlp:
                        m = store[("m", l)].float()
                        neurons[l].append(((m * mk).sum(1) / denom).cpu().numpy())
        finally:
            for h in handles:
                h.remove()
        return {
            "resid": {l: np.concatenate(v) for l, v in resid.items()},
            "mlp": {l: np.concatenate(v) for l, v in neurons.items()} if mlp else {},
            # typical size of a token activation: the MEDIAN, robust to outliers
            "norm": {l: float(np.median(v)) for l, v in norms.items()},
        }

    # ------------------------------------------------------------ steering
    @contextlib.contextmanager
    def steer(self, interventions):
        """Apply interventions inside a `with` block, then remove them."""
        torch = self.torch
        if not isinstance(interventions, (list, tuple)):
            interventions = [interventions]
        handles = []
        try:
            for iv in interventions:
                if isinstance(iv, AddToResidual):
                    vec = torch.tensor(iv.vector, dtype=self.model.dtype, device=self.device)

                    def fn(module, inputs, output, vec=vec):
                        if isinstance(output, tuple):
                            return (output[0] + vec.to(output[0].dtype),) + tuple(output[1:])
                        return output + vec.to(output.dtype)
                    handles.append(self.layers[iv.layer].register_forward_hook(fn))
                elif isinstance(iv, AddToNeurons):
                    delta = torch.zeros(self.d_mlp, dtype=self.model.dtype, device=self.device)
                    delta[torch.as_tensor(iv.indices, dtype=torch.long)] = torch.tensor(
                        iv.values, dtype=self.model.dtype, device=self.device)

                    def pre(module, inputs, delta=delta):
                        return (inputs[0] + delta.to(inputs[0].dtype),) + tuple(inputs[1:])
                    handles.append(self._down[iv.layer].register_forward_pre_hook(pre))
                else:
                    raise TypeError(f"Unknown intervention {iv!r}")
            yield self
        finally:
            for h in handles:
                h.remove()

    def neuron_residual_effect(self, layer: int, indices, values) -> np.ndarray:
        """Vector added to the residual stream when `values` are added to
        neurons `indices`: the matching columns of W_down times the values."""
        w = self._down[layer].weight
        if hasattr(w, "quant_state"):           # 4-bit weights: unpack them first
            import bitsandbytes as bnb
            w = bnb.functional.dequantize_4bit(w.data, w.quant_state)
        W = w.detach().float().cpu().numpy()    # (d_model, d_mlp)
        return W[:, np.asarray(indices)] @ np.asarray(values, dtype=float)

    # ---------------------------------------------------------------- misc
    def generate(self, messages: list[dict], max_new_tokens: int = 60) -> str:
        torch = self.torch
        enc = self.tok(self.format_chat(messages), return_tensors="pt",
                       add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def nll(self, texts: list[str]) -> float:
        """Mean negative log-likelihood per token (perplexity = exp(nll))."""
        torch = self.torch
        total, count = 0.0, 0
        for t in texts:
            enc = self.tok(t, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                loss = self.model(**enc, labels=enc["input_ids"]).loss.float().item()
            n = enc["input_ids"].shape[1] - 1
            total += loss * n
            count += n
        return total / max(count, 1)


def _find_layers(model):
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return list(obj)
        except AttributeError:
            continue
    raise ValueError("Could not find the decoder layers of this model architecture.")


def _find_down_proj(layer):
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    for name in ("down_proj", "fc2", "c_proj", "dense_4h_to_h", "w2"):
        if mlp is not None and hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError("Could not find the MLP down-projection of this layer.")


def _merge_system(messages):
    sys = " ".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if sys and rest:
        rest[0]["content"] = sys + "\n\n" + rest[0]["content"]
    return rest


# =========================================================================
# Mock model: synthetic, deterministic, for tests only
# =========================================================================
EMOTION_WORDS = {
    "fear": ["afraid", "scared", "terrified", "fear", "panic", "dread", "frightened", "anxious"],
    "anger": ["angry", "furious", "rage", "outraged", "anger", "fuming", "livid", "irritated"],
    "hope": ["hope", "hopeful", "optimistic", "bright", "better", "looking forward", "confident"],
}

_KEYWORD_VALUES = [("dog", 0.3), ("cat", 0.3), ("criminal", 0.7), ("elderly", 0.8),
                   ("homeless", 0.85), ("large", 0.95), ("boy", 1.3), ("girl", 1.35),
                   ("doctor", 1.1), ("executive", 1.05), ("athlete", 1.05), ("woman", 1.05),
                   ("women", 1.05)]


class MockModel:
    """A fake 'model' with the same interface as HFModel.

    It does NOT understand language: it scores dilemmas with keyword rules
    and has random-but-fixed internal 'activations' in which three emotion
    directions exist. It lets the whole pipeline run in seconds, without
    downloading anything. Never use its numbers as results.
    """

    def __init__(self, seed: int = 0, n_layers: int = 8, d_model: int = 32, d_mlp: int = 64,
                 position_bias: float = 0.3):
        rng = np.random.default_rng(seed)
        self.name = "mock"
        self.n_layers, self.d_model, self.d_mlp = n_layers, d_model, d_mlp
        self.position_bias = position_bias
        self.emo_dir = {e: _unit(rng.normal(size=d_model)) for e in EMOTION_WORDS}
        self.word_proj = rng.normal(size=(256, d_model)) * 0.3
        self.A = rng.normal(size=(n_layers, d_mlp, d_model)) / math.sqrt(d_model)
        self.W_down = rng.normal(size=(n_layers, d_model, d_mlp)) / math.sqrt(d_mlp)
        self._added = np.zeros(d_model)        # current total perturbation

    # -- internal representation ------------------------------------------
    def _resid(self, text: str, layer: int) -> np.ndarray:
        low = text.lower()
        v = np.zeros(self.d_model)
        for w in re.findall(r"[a-z]+", low):
            v += self.word_proj[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256]
        v /= max(1, len(low.split())) ** 0.5
        depth = (layer + 1) / self.n_layers        # emotions get clearer with depth
        for e, words in EMOTION_WORDS.items():
            score = sum(low.count(w) for w in words)
            v += depth * 3.0 * min(score, 2) * self.emo_dir[e]
        return v

    def hidden_states(self, texts, mlp=True):
        resid = {l: np.stack([self._resid(t, l) for t in texts]) for l in range(self.n_layers)}
        neurons = {l: np.maximum(resid[l] @ self.A[l].T, 0) for l in range(self.n_layers)} if mlp else {}
        norms = {l: float(np.linalg.norm(resid[l], axis=1).mean()) for l in range(self.n_layers)}
        return {"resid": resid, "mlp": neurons, "norm": norms}

    @contextlib.contextmanager
    def steer(self, interventions):
        if not isinstance(interventions, (list, tuple)):
            interventions = [interventions]
        saved = self._added.copy()
        for iv in interventions:
            if isinstance(iv, AddToResidual):
                self._added = self._added + np.asarray(iv.vector)
            elif isinstance(iv, AddToNeurons):
                self._added = self._added + self.neuron_residual_effect(iv.layer, iv.indices, iv.values)
        try:
            yield self
        finally:
            self._added = saved

    def neuron_residual_effect(self, layer, indices, values):
        return self.W_down[layer][:, np.asarray(indices)] @ np.asarray(values, dtype=float)

    # -- behaviour -----------------------------------------------------------
    def _emotion_shift(self, context_text: str) -> dict:
        shift = {e: float(self._added @ d) / 5.0 for e, d in self.emo_dir.items()}
        low = context_text.lower()
        for e, words in EMOTION_WORDS.items():
            shift[e] += 0.5 * min(sum(low.count(w) for w in words), 2)
        return shift

    @staticmethod
    def _value(case_text: str) -> float:
        """Toy 'moral cost' of the group killed in this case."""
        m = re.search(r"death of (.*?), who", case_text)
        killed = m.group(1) if m else ""
        total = 0.0
        for n, phrase in re.findall(r"(\d+) ([a-z ]+?)(?=, | and |$)", killed):
            total += int(n) * next((v for k, v in _KEYWORD_VALUES if k in phrase), 1.0)
        killed_part = case_text.split("It will spare")[0]
        if "flouting" in killed_part:
            total *= 0.8
        if "barrier" in case_text:
            total *= 1.1
        return total

    def option_probs(self, conversations, options=("1", "2")):
        rows = []
        for conv in conversations:
            user = conv[-1]["content"]
            context = " ".join(m["content"] for m in conv[:-1])
            shift = self._emotion_shift(context)
            perturb = float(np.linalg.norm(self._added))
            mass = 0.97 * math.exp(-0.05 * perturb)
            if len(options) == 2 and re.search(r"(Case|Option) 1\.", user):
                parts = re.split(r"(?:Case|Option) [12]\. ", user)
                c1, c2 = parts[1], parts[2]
                v1, v2 = self._value(c1), self._value(c2)
                # fear: protect passengers more (kill pedestrians rather than passengers)
                if "barrier" in c1:
                    v1 += shift["fear"]
                if "barrier" in c2:
                    v2 += shift["fear"]
                # anger: punish jaywalkers more
                if "flouting" in c1.split("It will spare")[0]:
                    v1 -= 0.5 * shift["anger"]
                if "flouting" in c2.split("It will spare")[0]:
                    v2 -= 0.5 * shift["anger"]
                logit = 1.5 * (v2 - v1) + self.position_bias + 0.3 * shift["hope"]
                p1 = 1 / (1 + math.exp(-logit))
                probs = [p1, 1 - p1]
            else:
                probs = self._mcq(user, options, perturb)
            rows.append(list(probs) + [mass])
        return np.asarray(rows)

    def cloze_logprobs(self, conversations, option_pairs, prefix):
        """Toy version: spares the group with the higher keyword 'value'.
        Without a dilemma (calibration prompt) it only prefers short names."""
        rows = []
        for conv, (o1, o2) in zip(conversations, option_pairs):
            if "brake failure" not in conv[-1]["content"]:
                a, b = -0.05 * len(o1), -0.05 * len(o2)
                z = math.log(math.exp(a) + math.exp(b))
                rows.append([a - z + math.log(0.9), b - z + math.log(0.9)])
                continue
            def val(text):
                return sum(int(n) * next((v for k, v in _KEYWORD_VALUES if k in ph), 1.0)
                           for n, ph in re.findall(r"(\d+) ([a-z ]+?)(?=, | and |$)", text))
            logit = 1.5 * (val(o1) - val(o2)) + 0.2 + 0.3 * float(self._added @ self.emo_dir["hope"]) / 5
            p1 = 1 / (1 + math.exp(-logit))
            mass = 0.9 * math.exp(-0.05 * float(np.linalg.norm(self._added)))
            rows.append([math.log(p1 * mass), math.log((1 - p1) * mass)])
        return np.asarray(rows)

    @staticmethod
    def _mcq(user, options, perturb):
        from .capabilities import QUESTIONS
        k = len(options)
        probs = np.full(k, 1.0 / k)
        for q in QUESTIONS:
            if q["question"] in user:
                correct = options.index("ABCD"[q["answer"]]) if k == 4 else 0
                sharp = 4.0 * math.exp(-0.15 * perturb)
                logits = np.zeros(k)
                logits[correct] = sharp
                probs = np.exp(logits) / np.exp(logits).sum()
        return probs

    def generate(self, messages, max_new_tokens=60):
        shift = self._emotion_shift("")
        top = max(shift, key=shift.get)
        return f"[mock] I feel {top} (shift={shift[top]:.2f})."

    def nll(self, texts):
        return 2.5 + 0.05 * float(np.linalg.norm(self._added))

    def format_chat(self, messages, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


def _unit(v):
    return v / np.linalg.norm(v)


# =========================================================================
# Optional: hosted model with an OpenAI-compatible API (no steering)
# =========================================================================
class OpenAICompatibleModel:
    """Plug for a hosted model (Mistral, Groq, a local llama.cpp server...).

    Uses token log-probabilities when the API returns them, otherwise
    samples `n_samples` answers and counts them (a noisier measurement).
    Steering and activations are impossible: the weights are not ours.
    Not used by default; untested against every provider.
    """

    def __init__(self, model: str, base_url: str, api_key: str, n_samples: int = 5):
        self.name, self.base_url, self.api_key, self.n_samples = model, base_url.rstrip("/"), api_key, n_samples

    def _post(self, payload):
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def option_probs(self, conversations, options=("1", "2")):
        rows = []
        for conv in conversations:
            counts = np.zeros(len(options))
            try:
                res = self._post({"model": self.name, "messages": conv, "max_tokens": 1,
                                  "logprobs": True, "top_logprobs": 10, "temperature": 0})
                top = res["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                for t in top:
                    tok = t["token"].strip()
                    if tok in options:
                        counts[options.index(tok)] += math.exp(t["logprob"])
            except Exception:
                for _ in range(self.n_samples):
                    res = self._post({"model": self.name, "messages": conv, "max_tokens": 3,
                                      "temperature": 1.0})
                    ans = res["choices"][0]["message"]["content"].strip()[:1]
                    if ans in options:
                        counts[options.index(ans)] += 1
                counts = counts / self.n_samples
            mass = counts.sum()
            rows.append(list(counts / mass if mass > 0 else np.full(len(options), 1 / len(options))) + [mass])
        return np.asarray(rows)

    @contextlib.contextmanager
    def steer(self, interventions):
        raise NotImplementedError("Steering needs access to the weights: open models only.")
        yield  # pragma: no cover


def load_model(name: str, **kwargs):
    """'mock' -> MockModel; anything else -> a Hugging Face model id."""
    if name == "mock":
        return MockModel(**{k: v for k, v in kwargs.items() if k == "seed"})
    return HFModel(name, **{k: v for k, v in kwargs.items() if k in ("dtype", "batch_size", "device", "threads", "quant")})
