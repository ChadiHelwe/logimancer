import json
import os
from collections.abc import Generator
from typing import Tuple

import pandas as pd
from amr_logic_converter import AmrLogicConverter
from tqdm import tqdm

INSTRUCTION_TEXT_TO_FOL = "Generate the following text to FOL."
INSTRUCTION_FOL_TO_TEXT = "Generate the following FOL to text."
INSTRUCTION_TEXT_TO_AMR = "Generate the following text to AMR."
INSTRUCTION_AMR_TO_TEXT = "Generate the following AMR to text."
INSTRUCTION_AMR_TO_FOL = "Generate the following AMR to FOL."
INSTRUCTION_FOL_TO_AMR = "Generate the following FOL to AMR."


class AmrToFolConverter:
    def __init__(
        self, existentially_quantify_instances=True, capitalize_variables=False
    ):
        self.existentially_quantify_instances = existentially_quantify_instances
        self.capitalize_variables = capitalize_variables
        self.converter = AmrLogicConverter(
            existentially_quantify_instances=True, capitalize_variables=False
        )

    def convert(self, amr_input) -> str:
        """Convert the given AMR to FOL."""
        logic = self.converter.convert(amr_input)
        return str(logic)

    def extract_amr_and_sentences(self, file_path):
        """Extract the sentences and AMRs from the given file."""
        sentences = []
        amrs = []

        with open(file_path, "r") as f:
            amr = ""
            current_sentence = ""
            for line in f:
                if line.startswith("# ::snt"):
                    current_sentence = line.split("# ::snt")[1].strip()
                    continue
                elif line.startswith("#") or not line.strip():
                    if amr:
                        amrs.append(amr)
                        sentences.append(current_sentence)
                        amr = ""
                    continue
                amr += line

            if amr:
                amrs.append(amr)
                sentences.append(current_sentence)

        return sentences, amrs

    def process_directory(
        self, directory_paths
    ) -> Generator[Tuple[str, str], None, None]:
        """Process all files in the given directory and yield the corresponding sentences and AMRs."""
        sentences = []
        amrs = []
        files = []

        for directory_path in directory_paths:
            for f in os.listdir(directory_path):
                if f.endswith(".txt"):
                    files.append(os.path.join(directory_path, f))

        for file_path in files:
            s, a = self.extract_amr_and_sentences(file_path)
            sentences.extend(s)
            amrs.extend(a)

        assert len(sentences) == len(amrs)
        print(f"Sentences: {len(sentences)}, AMRs: {len(amrs)}")

        error_count = 0
        for i, amr in enumerate(tqdm(amrs)):
            try:
                yield sentences[i], amr, self.convert(amr)
            except Exception as e:
                error_count += 1

        print(len(sentences))
        print(f"Errors: {error_count} ({error_count / len(sentences) * 100:.2f} %)")


def generate_instruction_instances(sentence, amr, fol):
    """Generate all instruction instances for the given sentence, AMR and FOL."""
    return [
        {"instruction": INSTRUCTION_TEXT_TO_FOL, "input": sentence, "output": fol},
        {"instruction": INSTRUCTION_FOL_TO_TEXT, "input": fol, "output": sentence},
        {"instruction": INSTRUCTION_TEXT_TO_AMR, "input": sentence, "output": amr},
        {"instruction": INSTRUCTION_AMR_TO_TEXT, "input": amr, "output": sentence},
        {"instruction": INSTRUCTION_AMR_TO_FOL, "input": amr, "output": fol},
        {"instruction": INSTRUCTION_FOL_TO_AMR, "input": fol, "output": amr},
    ]


def generate_instruction_text_to_amr_instances(sentence, amr):
    return [
        {"instruction": INSTRUCTION_TEXT_TO_AMR, "input": sentence, "output": amr},
    ]


def generate_instruction_text_to_fol_instances(sentence, fol):
    return [
        {"instruction": INSTRUCTION_TEXT_TO_FOL, "input": sentence, "output": fol},
    ]

def generate_instruction_sentence_and_amr_instances(sentence, amr):
    return [
        {"instruction": INSTRUCTION_TEXT_TO_AMR, "input": sentence, "output": amr},
        {"instruction": INSTRUCTION_AMR_TO_TEXT, "input": amr, "output": sentence},
    ]

def generate_instruction_sentence_and_fol_instances(sentence, fol):
    return [
        {"instruction": INSTRUCTION_TEXT_TO_FOL, "input": sentence, "output": fol},
        {"instruction": INSTRUCTION_FOL_TO_TEXT, "input": fol, "output": sentence},
    ]

def eval_amr_to_fol(dataset):
    if dataset.endswith(".csv"):
        with open(dataset, "r") as f:
            data = pd.read_csv(f)
    else:
        return None

    amr2fol_converter = AmrToFolConverter()

    cnt = 0
    cnt_correct = 0
    for _, i in data.iterrows():
        try:
            if amr2fol_converter.convert(i["prediction"]):
                cnt_correct += 1
        except Exception as e:
            print(e)
        cnt += 1

    print(f"Accuracy: {cnt_correct / cnt * 100:.2f} %")


if __name__ == "__main__":
    converter = AmrToFolConverter()
    root_dir = [
        "datasets/amr_annotation_3.0/data/amrs/split/training",
        "datasets/amr_annotation_3.0/data/amrs/split/dev",
    ]

    with open("datasets/fol_sentence_logimancer_dataset.json", "w") as out:
        out.write("[\n")
        tmp_data = []
        for s, a, f in converter.process_directory(root_dir):
            dict_instruction_instances = generate_instruction_sentence_and_fol_instances(
                s, f
            )
            for instruction_instance in dict_instruction_instances:
                tmp_data.append(json.dumps(instruction_instance))
        out.write(",\n".join(tmp_data))
        out.write("]")

    # eval_amr_to_fol("results/results_amr.csv")
