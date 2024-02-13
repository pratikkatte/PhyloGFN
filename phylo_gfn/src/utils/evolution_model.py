import numpy as np


class EvolutionModel(object):

    def __init__(self, evolution_cfg):

        self.prior_lambda = evolution_cfg.PRIOR_LAMBDA
        self.compute_prior = evolution_cfg.COMPUTE_PRIOR
        self.vocab_size = evolution_cfg.VOCAB_SIZE
        self.alpha = 1 / (self.vocab_size - 1)
        self.seq_length = evolution_cfg.SEQUENCE_LENGTH
        Q = self.alpha * np.ones((self.vocab_size, self.vocab_size))
        for i in range(self.vocab_size):
            Q[i, i] = -1

        D, U = np.linalg.eig(Q)
        U_inv = np.linalg.inv(U)
        self.jc_params = D, U, U_inv, Q

    def get_rate_matrix(self, t):
        """

        :param t: edge length
        :return:
        """
        D, U, U_inv, Q = self.jc_params
        return np.dot(U * np.exp(D * t), U_inv)

    def compute_log_prior_p(self, length):
        """
        compute prior p using exponential pdf
        :param length:
        :return:
        """
        log_prior_p = np.log(self.prior_lambda) - self.prior_lambda * length
        return log_prior_p

    def compute_partial_prob(self, data, at_root):
        """
        :param data: consists list of dict (probs, length)
        :param at_root: whether calculate probs at root, when at root, calculate scores without considering the root
        :return: partial prob for each nucleotide and total prob assuming a uniform distribution at the root
        """
        probs = np.ones_like(data[0][0])
        if len(probs.shape) != 1:
            probs = probs.reshape(-1, self.vocab_size)

        # at root, calculate scores without considering the root node
        if at_root:
            assert len(data) == 2
            p0, l0 = data[0]
            p1, l1 = data[1]
            l = l0 + l1
            transition_matrix = self.get_rate_matrix(l)
            p0 = np.dot(p0, transition_matrix)
            if self.compute_prior:
                log_prior_p = self.compute_log_prior_p(l)
                log_prior_p_per_nucleotide = log_prior_p/self.seq_length
                p0 = p0 * np.exp(log_prior_p_per_nucleotide)
            probs = np.multiply(probs, p0)
            probs = np.multiply(probs, p1)
        else:
            for p, l in data:
                if len(p.shape) != 1:
                    p = p.reshape(-1, self.vocab_size)
                transition_matrix = self.get_rate_matrix(l)
                p = np.dot(p, transition_matrix)
                if self.compute_prior:
                    log_prior_p = self.compute_log_prior_p(l)
                    log_prior_p_per_nucleotide = log_prior_p/self.seq_length
                    p = p * np.exp(log_prior_p_per_nucleotide)
                probs = np.multiply(probs, p)

        log_p = np.sum(np.log(np.sum(probs / self.vocab_size, axis=-1)))
        return probs, log_p
