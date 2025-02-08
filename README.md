Contact : clement.lion@saegus.com

# Evaluation Modele RAG

## Purpose of the repository

The purpose of this repository is to evaluate multiple RAG models in order to check their performance. The notebooks also give a method to quickly create a benchmark of questions / answers for a mission with clients. This can help validate technical choices.

**Important**: we evaluate a full RAG system, not an LLM or the ability of a LLM to retrieve an information from a piece of text. If you want to evaluate the former, you can use existing evaluations like this one: https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard, if you want to evaluate the latter, you can use datasets of questions and answers like the following: https://huggingface.co/datasets/nvidia/ChatRAG-Bench.

This repository is heavily inspired from this page : https://huggingface.co/learn/cookbook/rag_evaluation from huggingface.

Another interesting source is [this arxiv paper](https://arxiv.org/pdf/2408.08067) which defines a framework to evaluate RAG systems.

## Methodology

### Create a set of test questions

In order to evaluate a RAG system, we need a documentation on which to evaluate our model and a set of questions to do our evaluation. In our case, we use a set of wikipedia pages. Once we have retrieved our pages, we chunk them and we use a powerful LLM (in our case GPT-4) to generate a question and an answer from a subset of the chunks chosen at random.

These questions will be our test set.

In order to verify their corectness and relevance, we can either manually check them or use another LLM to evaluate these three criteria and discard questions that seem uninteresting in regard to our RAG.

### Choose the RAG system you want to evaluate

In order to create your RAG system, make sure that you put the documents with the answers in your documentation. You need to choose two components, your LLM and your index. In our case, we will evaluate the following architecures :
- Azure AI Search (with simple search) + GPT-3.5
- Azure AI Search (with hybrid search) + GPT-3.5
- Azure AI Search (with hybrid search and a few more "noise" documents containing no answers) + GPT-3.5
- Azure AI Search (with hybrid search and a few more "noise" documents containing no answers)  + Mistral Large
- GCP index (with hybrid search and a few more "noise" documents containing no answers) + gemini-1.5-flash-001

### Evaluate answers given by the RAG

Once we have validated our questions, we can ask them to our RAG system. The answers from the RAG system will be evaluated by an LLM (in our case GPT-4) and given a grade from 1 to 5 depending if it answers completely the question or in parts.

### First conclusions

With our different setups, we can make the fallowing hypothesis :
- Vector search is better when we search for text that is close to the question (not exaclty similar).
- Adding more documents make it more complicated to find the exact document that contains the information. That is why the "noisy" dataframes have more trouble making an answer.
- GCP has the best results for our subset of questions.

## Content

In the current repository, we have the following elements :
- Sample Data (a folder containing the texte of Wikipedia pages used to do the first tests)
- comparison.csv (a csv containing 10 questions and the answers provided by the different RAG systems)
- 01_data_retrieval.ipynb (A notebook to retrieve documents from wikipedia using the wikipedia API)
- 02_generate_sample_questions.ipynb (A notebook to generate questions and answers using a GPT-4 model)
- 03_evaluation_rag_aoai.ipynb (A notebook to use RAG in AOAI to answer the questions generated in 02_generate_sample_questions.ipynb)
- 04_evaluation_method_rag_gcp.ipynb (A notebook to use RAG in GCP to answer the questions generated 02_generate_sample_questions.ipynb)
- 05_evaluation_retriever.ipynb (A notebook to evaluate the retriever and try to evaluate the retriever of a RAG system)

## Running the repository

In order to run this repository, you will need to add a .env file containing the following credentials :

-- ID for Azure OpenAI --
- OPENAI_API_ENDPOINT
- OPENAI_API_VERSION
- OPENAI_API_KEY
- OPENAI_EMBEDDING_MODEL
- SEARCH_ENDPOINT
- SEARCH_KEY
- SEARCH_INDEX

-- ID for GCP --
- PROJECT_ID (ID of the project to bill for the operations)
- BUCKET_NAME (name of the bucket to use in GCP to store embeddings)
- GOOGLE_APPLICATION_CREDENTIALS (directory where gcp credentials are stored)

-- ID for Mistal --
- MISTRAL_URL
- MISTRAL_KEY

## Limits and Critique

- In the LLM as a judge configuration, a model will be more lenient towards model from the same constructor (GPT-4 is more lenient towards OpenAI models).
- The grades from 1 to 5 given to each response are not always relevant (it is hard to define a 3/5 response and a 4/5 response). It can be more interesting to see it as 1 = wrong answer, 5 = right answer and a grade between 2 and 4 is a flawed answer.
- A fine-tuned model for evaluation may have better results. We however don't have the means of training one.
- A lot of the questions created in our question database are questions where you either find the right answer or don't (no in between).

## Documentation

Chatbot Arena :
- https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

LLM as judge:
- https://huggingface.co/learn/cookbook/rag_evaluation
- https://eugeneyan.com/writing/llm-evaluators/
- https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method

Other sources on RAG Evaluation:
- https://arxiv.org/pdf/2408.08067
- https://huggingface.co/datasets/nvidia/ChatRAG-Bench
- https://www.rungalileo.io/hallucinationindex

Evaluation datasets:
- https://huggingface.co/datasets/mrqa-workshop/mrqa
- https://microsoft.github.io/msmarco/
- https://huggingface.co/datasets/hotpotqa/hotpot_qa

Interesting packages & tools:
- https://dev.to/guybuildingai/-top-5-open-source-llm-evaluation-frameworks-in-2024-98m

## Contributions

If you want to help, there are plenty of LLM models that need to be evaluated and plenty of advanced RAG techniques that can be used to improve results. Feel free to benchmark new methods.

We also need to find interesting datasets in order to do more RAG.