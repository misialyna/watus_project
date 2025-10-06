from dataclasses import dataclass
from supervision.metrics.mean_average_precision import MeanAveragePrecision
import torch

import numpy as np
import supervision as sv

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def _normalize_targets(self, targets):
        """
        Zwraca listę słowników per-obraz: [{"boxes":..., "class_labels":..., "size":(h,w)}, ...]
        Obsługuje:
        - listę list dictów (batche)
        - listę dictów (pojedynczy batch)
        - dict -> wartości to listy/ndarray-e (zagregowane przez Trainer)
        """
        if isinstance(targets, dict):
            n = len(next(iter(targets.values())))
            return [{k: targets[k][i] for k in targets} for i in range(n)]
        if isinstance(targets, (list, tuple)) and targets:
            if isinstance(targets[0], dict):
                return list(targets)  # pojedynczy batch
            if isinstance(targets[0], (list, tuple)):
                # lista batchy -> spłaszcz
                flat = []
                for b in targets:
                    flat.extend(b)
                return flat
        return []  # pusty przypadek awaryjny

    def _to_numpy(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        try:
            return np.array(x)
        except Exception:
            return None

    def _ensure_batch_dim(self, arr):
        if arr is None:
            return None
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            return arr[None, ...]  # [Q, C] -> [1, Q, C]
        if arr.ndim == 1 and arr.shape[-1] == 4:
            return arr[None, None, ...]  # [4] -> [1,1,4]
        return arr

    def _pick_pair_from_sequence(self, seq):
        """
        Szuka PARY (logits, boxes) takich, że:
          - boxes ma ostatni wymiar == 4
          - logits ostatni wymiar != 4
          - kształty są zgodne na wszystkich wymiarach z wyjątkiem ostatniego
            (czyli [B,Q,*] i [B,Q,4]).
        Zwraca (logits_np, boxes_np) albo (None, None).
        """
        # zbierz tylko tensory/tablice
        arrs = []
        for x in seq:
            if hasattr(x, "shape"):
                a = self._to_numpy(x)
                if a is not None:
                    arrs.append(a)

        # kandydaci
        boxes_cands = [a for a in arrs if a.ndim >= 2 and a.shape[-1] == 4]
        logit_cands = [a for a in arrs if a.ndim >= 2 and a.shape[-1] != 4]

        # dopasuj pary o tym samym shape bez ostatniego wymiaru
        def core_shape(a):  # shape bez ostatniego wymiaru
            return a.shape[:-1]

        best = None
        for b in boxes_cands:
            for l in logit_cands:
                if core_shape(b) == core_shape(l):
                    # preferuj większe Q (ale spójne)
                    score = core_shape(b)
                    best = (l, b)
                    # możemy od razu zwrócić pierwszą spójną parę
                    return best
        return (None, None)

    def _normalize_predictions(self, predictions):
        """
        Zwraca (logits_np [N,Q,C], boxes_np [N,Q,4]) – *tylko główne* wyjście.
        Pomija aux, straty itd.
        """
        # 1) Słownik – najprościej i najlepiej
        if isinstance(predictions, dict):
            logits = self._to_numpy(predictions.get("logits", None))
            boxes = self._to_numpy(predictions.get("pred_boxes", predictions.get("boxes", None)))
            logits = self._ensure_batch_dim(logits)
            boxes = self._ensure_batch_dim(boxes)
            return logits, boxes

        # 2) Lista/tupla – interpretuj jako listę batchy
        if isinstance(predictions, (list, tuple)) and len(predictions) > 0:
            first = predictions[0]

            # 2a) Każdy element to dict/tupla z wieloma polami
            if isinstance(first, (list, tuple, dict)):
                logits_list, boxes_list = [], []

                for batch_out in predictions:
                    if isinstance(batch_out, dict):
                        l = self._to_numpy(batch_out.get("logits", None))
                        b = self._to_numpy(batch_out.get("pred_boxes", batch_out.get("boxes", None)))
                        # jeśli brak kluczy, spróbuj heurystyki na sekwencji wartości
                        if l is None or b is None:
                            l2, b2 = self._pick_pair_from_sequence(list(batch_out.values()))
                            l = l if l is not None else l2
                            b = b if b is not None else b2
                    else:
                        l, b = self._pick_pair_from_sequence(batch_out)

                    l = self._ensure_batch_dim(l)
                    b = self._ensure_batch_dim(b)

                    # tylko jeśli obie istnieją i są spójne w [B,Q,*]
                    if l is not None and b is not None and l.shape[:-1] == b.shape[:-1]:
                        logits_list.append(l)
                        boxes_list.append(b)
                    # inaczej – pomijamy ten element

                if len(logits_list) == 0:
                    # puste bezpieczniki
                    return np.zeros((0, 1, 1), dtype=np.float32), np.zeros((0, 1, 4), dtype=np.float32)

                # W TYM MIEJSCU wszystkie elementy mają zgodne [B,Q,*], więc concat zadziała
                logits = np.concatenate(logits_list, axis=0)
                boxes = np.concatenate(boxes_list, axis=0)
                return logits, boxes

            # 2b) Pojedyncza krotka/tablica – użyj heurystyki
            l, b = self._pick_pair_from_sequence(predictions)
            l = self._ensure_batch_dim(self._to_numpy(l))
            b = self._ensure_batch_dim(self._to_numpy(b))
            return l, b

        # 3) Fallback
        return np.zeros((0, 1, 1), dtype=np.float32), np.zeros((0, 1, 4), dtype=np.float32)

    # --- wcześniej były 3 metody; uprośćmy je do wersji "pojedynczej listy" ---

    def _post_process_targets(self, targets_list):
        """targets_list: lista dictów per-obraz"""
        out = []
        for t in targets_list:
            h, w = t["size"]
            boxes = sv.xcycwh_to_xyxy(t["boxes"]) * np.array([w, h, w, h])
            boxes = torch.tensor(boxes)
            labels = torch.tensor(t["class_labels"])
            out.append({"boxes": boxes, "labels": labels})
        return out

    def _post_process_predictions(self, logits, boxes, sizes_hw):
        """
        logits: np.ndarray [N, Q, C]
        boxes:  np.ndarray [N, Q, 4] (znormalizowane 0..1)
        sizes_hw: torch.LongTensor [N,2] (H,W)
        """
        results = []
        N = sizes_hw.shape[0]
        for i in range(N):
            output = ModelOutput(
                logits=torch.from_numpy(logits[i:i + 1]),
                pred_boxes=torch.from_numpy(boxes[i:i + 1]),
            )
            post = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=sizes_hw[i:i + 1]
            )
            results.extend(post)  # 1-elementowa lista
        return results

    @torch.no_grad()
    def __call__(self, evaluation_results):
        # 0) Wejście z EvalPrediction
        predictions = evaluation_results.predictions
        targets = evaluation_results.label_ids

        # 1) Ujednolicenie struktur
        targets_list = self._normalize_targets(targets)  # lista per-obraz
        logits, pred_boxes = self._normalize_predictions(predictions)  # zagregowane [N,...]

        # 2) Rozmiary obrazów (H, W) jako tensor [N,2]
        sizes_hw = torch.tensor(np.array([t["size"] for t in targets_list]), dtype=torch.long)

        # 3) Post-process
        post_targets = self._post_process_targets(targets_list)  # list[{"boxes","labels"}]
        post_preds = self._post_process_predictions(logits, pred_boxes,
                                                    sizes_hw)  # list[{"boxes","scores","labels"}]

        # 3.1) Konwersja do sv.Detections
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        sv_targets = []
        for t in post_targets:
            boxes = to_np(t.get("boxes", np.zeros((0, 4), dtype=np.float32))).astype(np.float32)
            labels = to_np(t.get("labels", np.zeros((0,), dtype=np.int64))).astype(np.int64)
            sv_targets.append(sv.Detections(xyxy=boxes, class_id=labels))  # confidence nieobowiązkowe

        sv_preds = []
        for p in post_preds:
            boxes = to_np(p.get("boxes", np.zeros((0, 4), dtype=np.float32))).astype(np.float32)
            scores = to_np(p.get("scores", np.zeros((0,), dtype=np.float32))).astype(np.float32)
            labels = to_np(p.get("labels", np.zeros((0,), dtype=np.int64))).astype(np.int64)
            sv_preds.append(sv.Detections(xyxy=boxes, class_id=labels, confidence=scores))

        # 4) mAP (Supervision)
        map_metric = MeanAveragePrecision()  # domyślnie BOXES, class_agnostic=False
        map_result = map_metric.update(sv_preds, sv_targets).compute()

        # 5) Budowa słownika metryk
        metrics = {
            "map_50_95": round(float(map_result.map50_95), 4),
            "map_50": round(float(map_result.map50), 4),
            "map_75": round(float(map_result.map75), 4),
        }

        # 6) Per-klasa: uśrednienie AP po progach IoU (0.50–0.95)
        # matched_classes: np.ndarray z ID klas odpowiadających rzędom w ap_per_class
        ap_per_class = getattr(map_result, "ap_per_class", None)
        matched_cls = getattr(map_result, "matched_classes", None)

        if ap_per_class is not None and matched_cls is not None and len(matched_cls) > 0:
            # ap_per_class ma shape (num_matched_classes, num_iou_thresholds)
            ap_mean = ap_per_class.mean(axis=1)  # AP50:95 na klasę
            for cls_id, cls_ap in zip(matched_cls.tolist(), ap_mean.tolist()):
                cls_name = self.id2label.get(int(cls_id), int(cls_id)) if getattr(self, "id2label",
                                                                                  None) else int(cls_id)
                metrics[f"map_{cls_name}"] = round(float(cls_ap), 4)

        # 7) Zwróć
        return metrics