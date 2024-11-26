import { DataAPIClient } from "@datastax/astra-db-ts"
import { PuppeteerWebBaseLoader } from "langchain/document_loaders/web/puppeteer"
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import OpenAI from "openai"
import "dotenv/config" 
// import { SimilarityMetric } from "langchain/vectorstores/rockset"

const { 
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION, 
    ASTRA_DB_API_ENDPOINT, 
    ASTRA_DB_APPLICATION_TOKEN, 
    OPENAI_API_KEY
} = process.env

const openai = new OpenAI({ apiKey: OPENAI_API_KEY})

const f1_data = [
    'https://en.wikipedia.org/wiki/Formula_One',
    'https://www.formula1.com/en/latest/all',
    'https://en.wikipedia.org/wiki/2022_Formula_One_World_Championship',
    'https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship',
    'https://en.wikipedia.org/wiki/2024_Formula_One_World_Championship',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_drivers',
    'https://www.formula1.com/en/results/2022/races',
    'https://www.formula1.com/en/results/2023/races',
    'https://www.formula1.com/en/results/2024/races',
]

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
const db = client.db(ASTRA_DB_API_ENDPOINT, { namespace: ASTRA_DB_NAMESPACE})

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100
})

type SimilarityMetric = "dot_product" | "cosine" | "euclidean"

const createCollection = async (similarityMetric: SimilarityMetric = "dot_product") => {
    const res = await db.createCollection(ASTRA_DB_COLLECTION, {
        vector: {
            dimension: 1536,
            metric: similarityMetric
        }
    })
    console.log(res)
}

const loadSampleData = async () => {
    const collection = await db.collection(ASTRA_DB_COLLECTION)
    for await (const url of f1_data) {
        const content = await scrapePage(url)
        const chunks = await splitter.splitText(content)
        for await (const chunk of chunks) {
            const embedding = await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: chunk,
                encoding_format: "float"
            })
            
            const vector = embedding.data[0].embedding

            const res = await collection.insertOne({
                $vector: vector,
                text: chunk
            })
            console.log(res)
        }
    }
}

const scrapePage = async (url: string) => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
            headless: true
        },
        gotoOptions: {
            waitUntil: "domcontentloaded"
        },
        evaluate: async (page, browser) => {
            const result = await page.evaluate(() => document.body.innerHTML)
            await browser.close()
            return result
        }
    })
    return (await loader.scrape())?.replace(/<[^>]*>?/gm, '')
}

createCollection().then(() => loadSampleData())
