# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
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
# ==============================================================================

from openai import OpenAI


def get_gpt_response(message, model="gpt-4-1106-preview"):
    client = OpenAI()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role" : "user",
                "content": message,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content


def node_classification(batch):
  message = "This is a directed subgraph of arxiv citation network with " + str(batch.x.shape[0]) + " nodes numbered from 0 to " + str(batch.x.shape[0]-1) + ".\n"
  message += "The subgraph has " + str(batch.edge_index.shape[1]) + " edges.\n"
  for i in range(1, batch.x.shape[0]):
    feature_str = ','.join(f'{it:.3f}' for it in batch.x[i].tolist())
    message += "The feature of node " + str(i) + " is [" + feature_str + "] "
    message += "and the node label is " + str(batch.y[i].item()) + ".\n"
  message += "The edges of the subgraph are " + str(batch.edge_index.T.tolist()) + ' where the first number indicates source node and the second destination node.\n'
  message += "Question: predict the label for node 0, whose feature is [" + ','.join(f'{it:.3f}' for it in batch.x[0].tolist()) + "]. Give the label only and don't show any reasoning process.\n\n"

  return message


def link_prediction(batch, titles, reason=False):
  message = "This is a directed subgraph of arxiv citation network with " + str(batch.x.shape[0]) + " nodes numbered from 0 to " + str(batch.x.shape[0]-1) + ".\n"
  graph = batch.edge_index.T.unique(dim=0).tolist()
  message += "The titles of each paper:\n"
  for i in range(batch.x.shape[0]):
    message += "node " + str(i) + " is '" + titles[i][0] + "'\n" 
  message += "The sampled subgraph of the networks are " + str(graph) + ' where the first number indicates source node cites the second destination node.\n'
  message += "Hint: the direction of the edge can indicate some information of temporality.\n"
  message += "\nAccording to principle of the construction of citation networks and the given subgraph structure, answer the following questions:\n"
  message += "Question 1: predict whether there tends to form an edge "+str(batch.edge_label_index.T.tolist()[1])+".\n"
  message += "Question 2: predict whether there tends to form an edge "+str(batch.edge_label_index.T.tolist()[3])+".\n"
  message += "Question 3: predict whether there tends to form an edge "+str(batch.edge_label_index.T.tolist()[2])+".\n"
  message += "Question 4: predict whether there tends to form an edge "+str(batch.edge_label_index.T.tolist()[0])+".\n"
  if reason:
    message += "Answer yes or no and show reasoning process.\n\n"
  else:
    message += "Answer yes or no and don't show any reasoning process.\n\n"

  return message
