import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { AIMessage } from "@langchain/core/messages";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableWithMessageHistory } from "@langchain/core/runnables"
import type { BaseMessage } from "@langchain/core/messages";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";

const model = new ChatAnthropic({
  model: "claude-3-5-sonnet-20240620",
  temperature: 0
});

await model.invoke([
  new HumanMessage({ content: "Hi! I'm Bob" }),
  new AIMessage({ content: "Hello Bob! How can I assist you today?" }),
  new HumanMessage({ content: "What's my name?" }),
]);
// console.log(reply.content)


// const messageHistories: Record<string, InMemoryChatMessageHistory> = {};

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful assistant who remembers all details the user shares with you.`,
  ],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
]);

const chain = prompt.pipe(model);

// const withMessageHistory = new RunnableWithMessageHistory({
//   runnable: chain,
//   getMessageHistory: async (sessionId) => {
//     if (messageHistories[sessionId] === undefined) {
//       messageHistories[sessionId] = new InMemoryChatMessageHistory();
//     }
//     return messageHistories[sessionId];
//   },
//   inputMessagesKey: "input",
//   historyMessagesKey: "chat_history",
// });

// const config = {
//   configurable: {
//     sessionId: "abc2",
//   },
// };

// const response = await withMessageHistory.invoke(
//   {
//     input: "Hi! I'm Bob",
//   },
//   config
// );

// response.content;
// console.log(response.content)

// const followupResponse = await withMessageHistory.invoke(
//   {
//     input: "What's my name?",
//   },
//   config
// );

// followupResponse.content;
// console.log(followupResponse.content)

// const config2 = {
//   configurable: {
//     sessionId: "abc3",
//   },
// };

// const response2 = await withMessageHistory.invoke(
//   {
//     input: "What's my name?",
//   },
//   config2
// );

// response2.content;
// console.log(response2.content)

// const config3 = {
//   configurable: {
//     sessionId: "abc2",
//   },
// };

// const response3 = await withMessageHistory.invoke(
//   {
//     input: "What's my name?",
//   },
//   config3
// );

// response3.content;
// console.log(response3.content)

// type ChainInput = {
//   chat_history: BaseMessage[];
//   input: string;
// };

// const filterMessages = (input: ChainInput) => input.chat_history.slice(-10);

// const chain2 = RunnableSequence.from<ChainInput>([
//   RunnablePassthrough.assign({
//     chat_history: filterMessages,
//   }),
//   prompt,
//   model,
// ]);

const messages = [
  new HumanMessage({ content: "hi! I'm bob" }),
  new AIMessage({ content: "hi!" }),
  new HumanMessage({ content: "I like vanilla ice cream" }),
  new AIMessage({ content: "nice" }),
  new HumanMessage({ content: "whats 2 + 2" }),
  new AIMessage({ content: "4" }),
  new HumanMessage({ content: "thanks" }),
  new AIMessage({ content: "No problem!" }),
  new HumanMessage({ content: "having fun?" }),
  new AIMessage({ content: "yes!" }),
  new HumanMessage({ content: "That's great!" }),
  new AIMessage({ content: "yes it is!" }),
];

// const response4 = await chain2.invoke({
//   chat_history: messages,
//   input: "what's my name?",
// });
// response4.content;
// console.log(response4.content)

// const response5 = await chain2.invoke({
//   chat_history: messages,
//   input: "what's my fav ice cream",
// });
// response5.content;
// console.log(response5.content)


const messageHistories2: Record<string, InMemoryChatMessageHistory> = {};

const withMessageHistory2 = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: async (sessionId) => {
    if (messageHistories2[sessionId] === undefined) {
      const messageHistory = new InMemoryChatMessageHistory();
      await messageHistory.addMessages(messages);
      messageHistories2[sessionId] = messageHistory;
    }
    return messageHistories2[sessionId];
  },
  inputMessagesKey: "input",
  historyMessagesKey: "chat_history",
});

const config4 = {
  configurable: {
    sessionId: "abc4",
  },
};

const response7 = await withMessageHistory2.invoke(
  {
    input: "whats my name?",
  },
  config4
);

response7.content;
// console.log(response7.content)

const response8 = await withMessageHistory2.invoke(
  {
    input: "whats my favorite ice cream?",
  },
  config4
);

response8.content;
// console.log(response8.content)

const config5 = {
  configurable: {
    sessionId: "abc6",
  },
};

const stream = await withMessageHistory2.stream(
  {
    input: "hi! I'm todd. tell me a joke",
  },
  config5
);

for await (const chunk of stream) {
  console.log("|", chunk.content);
}