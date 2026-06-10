# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ASCII byte-level tokenizer for nucleotide sequences."""

from typing import List, Union
import torch


class NucleotideASCIITokenizer:
    """
    ASCII byte-level tokenizer for nucleotide sequences (A, T, C, G).
    
    This tokenizer converts nucleotide characters to their ASCII values using ord(),
    as specified in the meeting requirements for LLAMA3 genomic data processing.
    """
    
    def __init__(self, vocab_size: int = 256):
        """
        Initialize the ASCII tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size (256 for full ASCII range)
        """
        self.vocab_size = vocab_size
        
        # Standard nucleotide characters and their ASCII values
        self.nucleotides = {'A': ord('A'), 'T': ord('T'), 'C': ord('C'), 'G': ord('G')}
        self.unknown_char = ord('N')  # Use N for unknown/ambiguous nucleotides
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
        # Special tokens for sequence boundaries
        self.special_tokens = {
            '<PAD>': self.pad_token_id,
            '<EOS>': self.eos_token_id, 
            '<BOS>': self.bos_token_id,
        }
    
    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a nucleotide sequence to token IDs using ASCII values.
        
        Args:
            sequence: Input nucleotide sequence string (e.g., "ATCGATCG")
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs (ASCII values)
        """
        if not isinstance(sequence, str):
            raise ValueError("Input must be a string")
            
        # Convert to uppercase and get ASCII values
        sequence = sequence.upper().strip()
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
            
        for char in sequence:
            if char in self.nucleotides:
                token_ids.append(self.nucleotides[char])
            elif char.isalpha():  # Handle other nucleotide codes (N, R, Y, etc.)
                token_ids.append(ord(char))
            # Skip non-alphabetic characters (spaces, numbers, etc.)
            
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
            
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs back to nucleotide sequence.
        
        Args:
            token_ids: List of token IDs or tensor
            
        Returns:
            Decoded nucleotide sequence string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        decoded_chars = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue  # Skip padding
            elif token_id == self.bos_token_id or token_id == self.eos_token_id:
                continue  # Skip special tokens
            elif 32 <= token_id <= 126:  # Valid ASCII printable range
                decoded_chars.append(chr(token_id))
            else:
                decoded_chars.append('N')  # Unknown token
                
        return ''.join(decoded_chars)
    
    def text_to_ids(self, text: str) -> List[int]:
        """
        Convert text to token IDs (compatibility with NeMo tokenizer interface).
        
        Args:
            text: Input nucleotide sequence
            
        Returns:
            List of token IDs
        """
        return self.encode(text, add_special_tokens=False)
    
    def ids_to_text(self, ids: List[int]) -> str:
        """
        Convert token IDs to text (compatibility with NeMo tokenizer interface).
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        return self.decode(ids)
    
    @property
    def eod(self) -> int:
        """End of document token ID (compatibility with NeMo)."""
        return self.eos_token_id
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
