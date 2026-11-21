"""Test MMD-Hamming de Model Equality Testing (fiche 2, section 4.4)."""
import numpy as np


PAD = -1


def pad(tokens, L):
    """Tronque / complète une réponse pour qu'elle fasse exactement L tokens."""
    tokens = list(tokens)[:L]
    return tokens + [PAD] * (L - len(tokens))


def hamming_gram(Y):
    """G[i, j] = proportion de positions où les réponses i et j ont le même token."""
    return (Y[:, None, :] == Y[None, :, :]).mean(axis=-1)


def mmd_prompt(G, is_p):
    """MMD sans biais pour UN prompt. is_p[i] = True si la réponse i vient de P."""
    a, b = np.flatnonzero(is_p), np.flatnonzero(~is_p)
    n, m = len(a), len(b)
    Gaa, Gbb = G[np.ix_(a, a)], G[np.ix_(b, b)]
    intra_p = (Gaa.sum() - np.trace(Gaa)) / (n * (n - 1))  # sans la diagonale
    intra_q = (Gbb.sum() - np.trace(Gbb)) / (m * (m - 1))
    inter = G[np.ix_(a, b)].mean()
    return intra_p + intra_q - 2 * inter


def mmd_test(DP, DQ, B=1000, seed=0):
    """DP, DQ : {prompt: array (n, L) de tokens}. Renvoie (MMD, p-value, nulle)."""
    rng = np.random.default_rng(seed)
    prompts = list(DP)
    G = {x: hamming_gram(np.concatenate([DP[x], DQ[x]])) for x in prompts}
    lab = {x: np.r_[np.ones(len(DP[x]), bool), np.zeros(len(DQ[x]), bool)]
           for x in prompts}

    def stat(labels):
        return np.mean([mmd_prompt(G[x], labels[x]) for x in prompts])

    obs = stat(lab)
    # test de permutation : on mélange les étiquettes P/Q À L'INTÉRIEUR de chaque prompt
    null = np.array([stat({x: rng.permutation(lab[x]) for x in prompts})
                     for _ in range(B)])
    p_value = (1 + np.sum(null >= obs)) / (B + 1)
    return obs, p_value, null
