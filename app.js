import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
// import { StringOutputParser } from "@langchain/core/output_parsers";
// import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { VoyageEmbeddings } from "@langchain/community/embeddings/voyage";
// import { Document } from "@langchain/core/documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
// import { MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

const chatModel = new ChatAnthropic({});

// // var answer = await chatModel.invoke("what is LangSmith?");

// // console.log(answer);

// const prompt = ChatPromptTemplate.fromMessages([
//   ["system", "You are a world class technical documentation writer."],
//   ["user", "{input}"],
// ]);

// const chain = prompt.pipe(chatModel);

// var answer = await chain.invoke({
//   input: "what is LangSmith?",
// });

// console.log(answer);

// const outputParser = new StringOutputParser();

// const llmChain = prompt.pipe(chatModel).pipe(outputParser);

// var answer = await llmChain.invoke({
//   input: "what is LangSmith?",
// });

// console.log(answer);

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/"
);

const docs = await loader.load();

// console.log(docs.length);
// console.log(docs[0].pageContent.length);

// const textSplitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 1000,
//   chunkOverlap: 200,
// });

// const splits = await textSplitter.splitDocuments(docs);

// console.log(splits.length);
// console.log(splits[0].pageContent);

const embeddings = new VoyageEmbeddings({
  apiKey: process.env.VOYAGEAI_API_KEY, 
});


// const vectorStore = await MemoryVectorStore.fromDocuments(
//   splits,
//   embeddings
// );

const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);
// console.log(splitDocs.length);

const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

// await documentChain.invoke({
//   input: "what is LangSmith?",
//   context: [
//     new Document({
//       pageContent:
//         "LangSmith is a platform for building production-grade LLM applications.",
//     }),
//   ],
// });

// console.log(reply)

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

const result = await retrievalChain.invoke({
  input: "what is LangSmith?",
});

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

// console.log(reply)