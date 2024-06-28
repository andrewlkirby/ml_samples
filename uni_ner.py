import torch
from transformers import pipeline
from typing import List, Dict, Union, Any
import re

from uni_ner_utils.utils import preprocess_instance

generator = pipeline('text-generation', 
                     model = "Universal-NER/UniNER-7B-type", 
                     torch_dtype = torch.float16, 
                     device = 'cuda' # I'm assuming you're using a CUDA GPU
                     )

entity_types = ["PERSON", 
                "ORG", 
                "LOCATION", 
                "PRODUCT_NAME", 
                "EVENT",
                "LANGUAGE",
                "DATE",
                "TIME",
                "PERCENT",
                "QUANTITY",
                "ORDINAL",
                "CARDINAL"
               ]

class UniversalNER:
    
    def __init__(self, input_text: str):
        self.input_text = input_text
        
    def run_universal_ner(self,
                          entity_type: str,
                          max_new_tokens: int = 256,
                          ) -> Dict[str, Union[str, List[str]]]:
        """
        `example` has our prompt that we use to get the output.
        """
        example = {"conversations": [{"from": "human", "value": f"Text: {self.input_text}"}, 
                                     {"from": "gpt", "value": "I've read this text."}, 
                                     {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                                     {"from": "gpt", "value": "[]"}
                                    ]
                  }
        prompt = preprocess_instance(example['conversations'])
        outputs = generator(prompt, 
                            max_length = max_new_tokens, 
                            return_full_text = False,
                            truncation = True
                           )
        return {"entity_type": entity_type, "entity": eval(outputs[0]['generated_text'])}

    def get_universal_ner(self) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Goes through each type of entity in `entity_types` and runs the model.
        """
        output_items = []
        for entity_type in entity_types:
            output = self.run_universal_ner(entity_type)
            if output['entity']: output_items.append(output)
        return output_items

    def get_itemized_output(self,
                            ner_output: List[Dict[str, Union[str, List[str]]]]
                            ) -> List[Dict[str, Any]]:
        """
        Since the outputs come in a list of entities for each entity type,
        we want to sort by and get spans for each individual entity.
        """
        itemized_output = []
        completed_entities = []
        for ner_item in ner_output:
            entity_type = ner_item['entity_type']
            entities = ner_item['entity']
            for entity in entities:
                if entities.count(entity) > 1:
                    if entity not in completed_entities:
                        for match in re.finditer(entity, self.input_text):
                            itemized_output.append({'entity_type': entity_type,
                                                    'entity': entity,
                                                    'span': match.span()
                                                    }
                                                    )
                            completed_entities.append(entity)
                else:
                    match = re.search(entity, self.input_text)
                    itemized_output.append({'entity_type': entity_type,
                                            'entity': entity,
                                            'span': match.span()
                                            }
                                            )
        return itemized_output

    def run(self) -> List[Dict[str, Any]]:
        ner_output = self.get_universal_ner()
        itemized_ner_output = self.get_itemized_output(ner_output)
        return itemized_ner_output