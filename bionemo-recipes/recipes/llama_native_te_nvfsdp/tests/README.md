# Unit Tests for LLAMA3 Genomic Dataset

## Test Structure

Testing principles:
- ✅ **Simple repeated characters** (AAAA, TTTT, CCCC, GGGG)
- ✅ **No parametrize** - explicit individual test functions
- ✅ **Test the logic, not complex data**
- ✅ **Clear assertions** - `expected = [65] * 10`

## Test Files

### `test_tokenizer.py` (9 tests)
Tests tokenizer interface and behavior:
- Special token IDs (NeMo convention)
- Encoding/decoding operations
- Padding and attention masks
- Nucleotide mappings

### `test_windowing.py` (6 tests)
Tests dataset observable behavior:
- Window creation from sequences
- Sample retrieval (__getitem__)
- Special token presence
- Deterministic behavior with seeds

### `test_dataloader.py` (7 tests)
Tests dataloader interface and contracts:
- Batch structure and keys
- Label correctness for causal LM
- Padding behavior
- Infinite iteration
- Deterministic with seeds

## Total: 22 tests (behavior-focused)

## Design Philosophy

Tests focus on **observable behavior** rather than implementation details:
- Test what the API returns, not how it computes it
- Avoid checking internal data structures (window_mappings, shuffled_indices)
- Focus on contracts: given inputs, verify output properties
- Makes tests robust to refactoring

## Detailed Tests (Backup)

For reference, detailed implementation-testing versions are saved as:
- `test_tokenizer_detailed.py`
- `test_windowing_detailed.py`
- `test_dataloader_detailed.py`

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_tokenizer.py -v
pytest tests/test_windowing.py -v
pytest tests/test_dataloader.py -v

# Run specific test
pytest tests/test_tokenizer.py::test_tokenizer_special_token_ids -v

# Run with verbose output
pytest tests/ -vv

# Run and stop at first failure
pytest tests/ -x

# Show test coverage
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Data

Uses dummy SQLite database with:
- `seq_A`: 10,000 As
- `seq_T`: 8,000 Ts
- `seq_C`: 5,000 Cs
- `seq_G`: 2,000 Gs

Makes test expectations obvious!

