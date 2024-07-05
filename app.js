import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createRetrieverTool } from "langchain/tools/retriever";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { AgentExecutor, createXmlAgent } from "langchain/agents";
import { PromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";

import 'dotenv/config'


const chatModel = new ChatAnthropic({});

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);

const docs = await loader.load();

const embeddings = new VoyageEmbeddings({
  apiKey: process.env.VOYAGEAI_API_KEY, 
});


const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs.length);

const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
// const prompt =
//   ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

// <context>
// {context}
// </context>

// Question: {input}`);

// const documentChain = await createStuffDocumentsChain({
//   llm: chatModel,
//   prompt,
// });


const retriever = vectorstore.asRetriever();

// const retrievalChain = await createRetrievalChain({
//   combineDocsChain: documentChain,
//   retriever,
// });

// const result = await retrievalChain.invoke({
//   input: "what is LangSmith?",
// });

// console.log(result.answer);

const historyAwarePrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  [
    "user",
    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
  ],
]);

const historyAwareRetrieverChain = await createHistoryAwareRetriever({
  llm: chatModel,
  retriever,
  rephrasePrompt: historyAwarePrompt,
});

const chatHistory = [
  new HumanMessage("Can LangSmith help test my LLM applications?"),
  new AIMessage("Yes!"),
];

await historyAwareRetrieverChain.invoke({
  chat_history: chatHistory,
  input: "Tell me how!",
});

const historyAwareRetrievalPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "Answer the user's questions based on the below context:\n\n{context}",
  ],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);

const historyAwareCombineDocsChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt: historyAwareRetrievalPrompt,
});

const conversationalRetrievalChain = await createRetrievalChain({
  retriever: historyAwareRetrieverChain,
  combineDocsChain: historyAwareCombineDocsChain,
});

const result2 = await conversationalRetrievalChain.invoke({
  chat_history: [
    new HumanMessage("Can LangSmith help test my LLM applications?"),
    new AIMessage("Yes!"),
  ],
  input: "tell me how",
});

console.log(result2.answer);

// throwing error
// const retrieverTool = await createRetrieverTool(retriever, {
//   name: "langsmith_search",
//   description:
//     "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
// });

// const tavilyApiKey = process.env.TAVILY_API_KEY

// const tools = [new TavilySearchResults({ maxResults: 1 })];


// Get the prompt to use - you can modify this!
// If you want to see the prompt in full, you can at:
// https://smith.langchain.com/hub/hwchase17/xml-agent-convo

// const prompt =
//   ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

// const prompt = await pull<PromptTemplate>("hwchase17/xml-agent-convo");

// const llm = new ChatAnthropic({
//   temperature: 0,
// });

// const agent = await createXmlAgent({
//   llm,
//   tools,
//   prompt
// });

// const agentExecutor = new AgentExecutor({
//   agent,
//   tools,
// });

// const result3 = await agentExecutor.invoke({
//   input: "what is LangChain?",
// });

// console.log(result3);