import os
import sys
import argparse
import json
import pprint
import logging
from collections import defaultdict
from typing import *

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from paq.paq_utils import load_jsonl, dump_jsonl

pp = pprint.PrettyPrinter(indent=2)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_answer_langs(answer_mode: str, valid_languages: Iterable[str], current_language: str) -> List[str]:
    if answer_mode == "bi" and current_language == "en":
        answer_mode = "en"
    
    if answer_mode == "multi":
        return valid_languages
    elif answer_mode == "en":
        return ["en"]
    elif answer_mode == "bi":
        return [current_language, "en"]
    elif answer_mode == "mono":
        return [current_language]
    else:
        raise ValueError(f"Answer mode {answer_mode} not recognised for current dataset")

def main(args: argparse.Namespace) -> None:
    mkqa_src = load_jsonl(args.source)
    valid_languages = sorted(mkqa_src[0]['queries'].keys())

    lang2q = defaultdict(list)    
    for ex in mkqa_src:
        for lang in valid_languages:
            lang_ex = {}

            # Question is always the ssaem
            lang_ex['question'] = ex['queries'][lang]
            lang_ex['answer'] = []

            for ans_lang in get_answer_langs(args.nlang, valid_languages, lang):
                lang_ex['answer'].extend([ans['text'] for ans in ex["answers"][ans_lang] if ans['text'] is not None])

                if args.use_alias:
                    lang_ex['answer'].extend([alias for ans in ex["answers"][ans_lang] for alias in ans.get('aliases', [])])

            lang_ex['answer'] = list(set(filter(lambda x: x is not None, lang_ex['answer'])))
            
            if lang_ex['answer'] != []:
                lang2q[lang].append(lang_ex)

    # Output
    for lang in valid_languages:
        output_path = f"{args.dest}/{lang}.jsonl"
        dump_jsonl(lang2q[lang], output_path)

    if args.combine:
        output_path = f"{args.dest}/all.jsonl"
        all2q = []
        for lang in valid_languages:
            all2q.extend(lang2q[lang])
        dump_jsonl(all2q, output_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help="Path to the MKQA source JSON file")
    parser.add_argument('--dest', type=str, help="Directory to save .jsonl formatted outputs")
    parser.add_argument('--combine', action="store_true", help="Dump all languages into a single file")
    parser.add_argument('--use_alias', action="store_true", help='Include aliases in correct answers')
    parser.add_argument('--nlang', choices=['mono', 'en', 'bi', 'multi'], default='bi', 
                        help="""
                            Language formatting for answers: 
                            [mono]: Only question language,
                            [en]: English only,
                            [bi]: Question language + English,
                            [multi]: All languages
                        """)
    args = parser.parse_args()
    main(args)
