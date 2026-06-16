# Notebook data for `NotebookViewer`

Place **generated** `.ts` modules here (one per notebook). They are produced by:

```bash
python fern/scripts/ipynb-to-fern-json.py path/to/notebook.ipynb -o fern/components/notebooks/my-notebook.json
```

The script writes both `my-notebook.json` and `my-notebook.ts`. Import the **`.ts`** file in MDX:

```mdx
import { NotebookViewer } from "@/components/NotebookViewer";
import notebook from "@/components/notebooks/my-notebook";

<NotebookViewer notebook={notebook} colabUrl="https://colab.research.google.com/github/NVIDIA/bionemo-framework/blob/main/..." />
```

Generated `*.ts` files may be gitignored or committed, depending on repo policy.
