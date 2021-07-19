from typing import List
import numpy as np
from vae.latents_explorer import LatentsExplorer
from socionics_db import get_weight


class LatentsExplorerPassthr(LatentsExplorer):
    def __post_init__(self):
        if self.y is not None:
            self.y = self.y + 1
        if self.y_s is not None:
            self.y_s = self.y_s + 1

    def override_w(self) -> None:
        assert self.y is not None
        types_sex, males = np.copy(self.y + 1), self.z[:, 0] > 0.5
        types_sex[males] += 16
        self.w = get_weight(types_sex)

    def get_lbl_stats(self, idxs_lbl_subset: List[int]) -> str:
        s_lbl = self.z[self.idxs_lbl][:, 0]
        assert self.y_lbl is not None

        csv, dic = self._get_lbl_stats(self.y_lbl, idxs_lbl_subset)
        assert len(dic) == 16
        csv_0, dic = self._get_lbl_stats(self.y_lbl, idxs_lbl_subset, mask_lbl=s_lbl < 0.5, mask_postfix='_female')
        assert len(dic) == 16
        csv_1, dic = self._get_lbl_stats(self.y_lbl, idxs_lbl_subset, mask_lbl=s_lbl >= 0.5, mask_postfix='_male')
        assert len(dic) == 16

        return csv + '\n\n' + csv_0 + '\n\n' + csv_1
