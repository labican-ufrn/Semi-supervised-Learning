from base_flexcons import BaseFlexCon

class FlexConC(BaseFlexCon):
    def __init__(self, base_classifier, cr=0.05, threshold=0.95, margin=0.01, max_iter=10, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, max_iter=max_iter, verbose=verbose)
        self.cr = cr
        self.margin = margin
        self.max_iter = max_iter

    def adjust_threshold(self, local_measure):
        # Calcula o `min_precision` dinamicamente com base no histórico de precisão
        dynamic_min_precision = self.get_dynamic_min_precision()
        
        if local_measure > (dynamic_min_precision + self.margin) and (self.threshold - self.cr) > 0.0:
            self.threshold -= self.cr
        elif local_measure < (dynamic_min_precision - self.margin) and (self.threshold + self.cr) <= 1:
            self.threshold += self.cr