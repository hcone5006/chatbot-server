import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { AIMessage } from "@langchain/core/messages";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableWithMessageHistory } from "@langchain/core/runnables"

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


const messageHistories: Record<string, InMemoryChatMessageHistory> = {};

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful assistant who remembers all details the user shares with you.`,
  ],
  ["placeholder", "{chat_history}"],
  ["human", "{input}"],
]);

const chain = prompt.pipe(model);

const withMessageHistory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: async (sessionId) => {
    if (messageHistories[sessionId] === undefined) {
      messageHistories[sessionId] = new InMemoryChatMessageHistory();
    }
    return messageHistories[sessionId];
  },
  inputMessagesKey: "input",
  historyMessagesKey: "chat_history",
});

const config = {
  configurable: {
    sessionId: "abc2",
  },
};

const response = await withMessageHistory.invoke(
  {
    input: "Hi! I'm Bob",
  },
  config
);

response.content;
console.log(response.content)