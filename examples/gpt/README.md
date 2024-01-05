# Using GPT to Reason on Graphs

This simple example shows how to leverage GPT to make inference on large graphs.

### 1. Prepare the graph dataset & environment
In this example, we use the dataset
[arxiv_2023](https://github.com/TRAIS-Lab/LLM-Structured-Data/tree/main/dataset/arxiv_2023)
and download it to the path `../data/arxiv_2023`.

Then, export your OPENAI_API_KEY as the environment variable in your shell:

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
```

### 2. Run the code
Configure the parameters in the data loader and run the code.
```bash
python arxiv.py
```
This example tests the inference performance of GPT on a large graph with the link prediction task as the default task. 

First, we sample a 2-hop ego-subgraph from the original graph. The subgraph is sampled by the `LinkNeighborLoader` with a mini-batch sampler that samples a fixed number of neighbors for each edge and is formed as PyG's `edge_index`.

Then, the sampled subgraph along with node features (e.g. in this example the title for the paper node) is fed into GPT to infer whether a requested edge is in the original graph or not. 


### Appendix:
1. **Dataset**: You can also use other datasets and modify the preprocessing code, but don't forget to transform the graph format into PyG's `edge_index`.
2. **Prompts**: Use the parameter `reason: Bool` to decide whether to see the reasoning process of GPT. You can also design your own prompts to make inference on graphs instead of using our template. 
3. **Node classification**: We also provide a template prompt for node classification task, and design your method to leverage the label informatiion. 
**Note**: We've tried directly passing the labels for nodes in an ego-subgraph to predict the label of the center node, and GPT's prediction behavior in this case is close to voting via neighbors.
