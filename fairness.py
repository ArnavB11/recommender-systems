import numpy as np

class FairnessObjective:
    """
    Computes the recommendation-list fairness objective:
    F(C) = mean(HDB(i) * GEP(i) * Quality(i)) for every movie i in C.
    """

    EPSILON = 1e-8

    def __init__(self, data):
        self.data = data
        self.max_rating_count = 0.0
        self.global_avg_rating = 0.0
        self.max_possible_rating = 5.0
        self.gep_scores = {}
        self.quality_scores = {}
        self._is_fit = False

    def fit(self):
        """
        Precompute fixed global exposure penalty and quality scores for every item.
        """
        item_popularity = self.data.item_popularity
        item_avg_rating = self.data.item_avg_rating

        self.max_rating_count = float(max(item_popularity.values(), default=0.0))
        self.global_avg_rating = (
            float(np.mean(list(item_avg_rating.values())))
            if item_avg_rating else 0.0
        )
        self.max_possible_rating = 5.0

        for item_idx in range(self.data.n_items):
            rating_count = float(item_popularity.get(item_idx, 0.0))
            avg_rating = float(item_avg_rating.get(item_idx, self.global_avg_rating))

            if self.max_rating_count == 0.0:
                gep = 1.0
            else:
                gep = 1.0 - (rating_count / self.max_rating_count)

            quality = avg_rating / self.max_possible_rating

            self.gep_scores[item_idx] = float(np.clip(gep, 0.0, 1.0))
            self.quality_scores[item_idx] = float(np.clip(quality, 0.0, 1.0))

        self._is_fit = True
        return self

    @staticmethod
    def normalize_to_unit(scores_dict):
        """
        Normalize item scores by the maximum score in the dictionary.
        """
        if not scores_dict:
            return {}

        max_score = max(float(score) for score in scores_dict.values())
        if max_score == 0.0:
            return dict(scores_dict)

        return {
            item_idx: float(np.clip(float(score) / max_score, 0.0, 1.0))
            for item_idx, score in scores_dict.items()
        }

    def compute_fairness(self, chromosome, dopm_scores_dict, serendipity_scores_dict):
        """
        Compute the mean fairness score for a recommendation chromosome.
        """
        if not chromosome:
            return 0.0

        if not self._is_fit:
            self.fit()

        normalized_dopm = self.normalize_to_unit(dopm_scores_dict)
        normalized_serendipity = self.normalize_to_unit(serendipity_scores_dict)
        prior_quality = self.global_avg_rating / self.max_possible_rating

        movie_scores = []
        for item_idx in chromosome:
            n_i = float(normalized_dopm.get(item_idx, 0.0))
            s_i = float(normalized_serendipity.get(item_idx, 0.0))
            hdb_i = (2.0 * n_i * s_i) / (n_i + s_i + self.EPSILON)
            gep_i = float(self.gep_scores.get(item_idx, 1.0))
            quality_i = float(self.quality_scores.get(item_idx, prior_quality))
            movie_scores.append(hdb_i * gep_i * quality_i)

        return float(np.mean(movie_scores))