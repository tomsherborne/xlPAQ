#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
import logging
import os
from paq.retrievers.embed import embed_job

logger = logging.getLogger(__name__)
CUDA = torch.cuda.is_available()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument('--qas_to_embed', type=str,required=True, help='Path to questions to embed in jsonl format')
    parser.add_argument('--n_jobs', type=int, required=True, help='how many jobs to embed with (n_jobs=-1 will run a single job locally)')
    parser.add_argument('--output_dir', type=str, help='path to write vectors to')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--memory_friendly_parsing', action='store_true', help='Pass this to load jsonl files more slowly, but save memory')
    parser.add_argument('--embed_job_n', type=int, required=True, help="Index of chunked embedding processor")
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.fp16 and not CUDA:
        raise Exception('Cant use --fp16 without a gpu, CUDA not found')

    output_path = os.path.join(args.output_dir, 'embeddings.pt')

    embed_job(
        args.qas_to_embed,
        args.model_name_or_path,
        output_path,
        args.n_jobs,
        args.embed_job_n,
        args.batch_size,
        args.fp16,
        args.memory_friendly_parsing
    )
