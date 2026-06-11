git ls-files | grep -vE '\.parquet$|\.pkl$' | while read file; do
  echo "===== FILE: $file ====="
  cat "$file"
  echo ""
done > gitingest.txt
