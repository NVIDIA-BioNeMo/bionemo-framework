from tqdm import tqdm
import numpy as np
import pandas as pd

def RLDIF_Generator(model, dataloader, num_samples = 4, foldfunction = None):

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    index_to_AA = {i: a for i, a in enumerate(alphabet)}

    results = []

    per_batch_accuracy = []

    per_batch_tm_vec_output = []
    per_batch_tm_scores = []

    for batch in tqdm(dataloader):

        # Sample 4 times
        for i in range(num_samples):
            out = model.sample(batch.clone().cuda(), closure=True)
            names = batch["names"]

            accs = []
            counter = 0
            num = 0
            for ft, fp, mask, name in zip(
                out["features_true"],
                out["features_0_step"],
                out["mask"],
                names,
            ):
                ft = ft[mask.astype(bool)]
                n = ft.shape[0]
                counter += n
                acc = (ft.argmax(axis=-1) == fp.argmax(axis=-1)).sum() / float(n)
                fp = fp.argmax(axis=-1)
                ft = ft.argmax(axis=-1)

                pred_sequence = "".join(
                    np.vectorize(index_to_AA.get)(fp).tolist()
                )
                real_sequence = "".join(
                    np.vectorize(index_to_AA.get)(ft).tolist()
                )

                accs.append(acc)
                #print(acc)

                if foldfunction is not None:
                    tm_scores = foldfunction(pred_sequence, real_sequence)
                    print(tm_scores)
                else:
                    tm_scores = None

                results.append(
                    {
                        "name": name,
                        "pred": pred_sequence,
                        "real": real_sequence,
                        "tm_score": tm_scores,
                    }
                )

                num += 1

        acc = np.mean(accs)
        print(f"Accuracy: {acc}")

        if tm_scores is not None:
            tm_scores = np.mean(tm_scores)
        else:
            print(f"TM-Score Output: {tm_scores}")
 
        per_batch_accuracy.append(acc)
        per_batch_tm_scores.append(tm_scores)

    print(f"Average Accuracy: {np.mean(per_batch_accuracy)}")
    if foldfunction is not None:
        print(f"Average TM-Score: {np.mean(per_batch_tm_scores)}")

    df = pd.DataFrame(results)
    return df 

