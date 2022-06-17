# -*- coding: utf-8 -*-
'''
This is an example for fine-tuning BERT for downstream tasks using Transformers by Huggingface (PyTorch).
Notes:
- Do note the difference between pre-training BERT, using BERT for encoding, and using BERT for fine-tuning.
'''

from torch import nn
from transformers.modeling_bert import BertModel, BertPreTrainedModel

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        linear = [nn.Linear(config.hidden_size, self.config.num_labels), nn.Sigmoid()]
        self.classifier = nn.Sequential(*linear)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
                            # inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # change loss function to BCELoss
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

