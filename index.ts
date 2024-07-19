import { readableStreamToText } from "bun";
import { type Application, type Request as eRequest, type Response as eResponse } from "express";
import express from 'express';
import path from "path";
import TypeDoc from "typedoc";
import { Builder, By, until, WebDriver } from 'selenium-webdriver';
import fs from "fs/promises"
import firefox from "selenium-webdriver/firefox"
const console = {
    log: async (...data: any[]) => {
        Bun.write(Bun.stdout, data+'\n')
    },
    error: async (...data: any[]) => {
        require("node:console").error(data)
    }
}
/**
 * @remarks Simple webserver to serve the documentation
 * @example 
 * ```ts
 * import { type Application } from "express";
 * docServer().then((Server: Application) => {
 *   Server.listen(8080);
 * })
 * ```
*/
const docServer = async (): Promise<Application> => {
    const app = await TypeDoc.Application.bootstrapWithPlugins({
        entryPoints: ["index.ts"],
    });
    const project = await app.convert();

    if (project) {
        const outputDir = "docs";

        await app.generateDocs(project, outputDir);
    }
    const s = express()
    s.use(express.static(path.join(__dirname, "docs")))
    return s;
}
/**
 * @remarks Web scraper function to get training data from the iDTech website.
 * @param firstID - Page ID to start at (defaults to 26753)
 * @param lastID - Page ID to stop at (defaults to 26852)
 * @param courseID - Course ID (defaults to 331)
 * @example 
 * ```ts
 * 
 * ```
 * @returns Returns `true` if success, `Error` if any failure
 */
const scraper = async (firstID: number = 26753, lastID: number = 27506, courseID: number = 331): Promise<boolean | Error> => {
    let currentID = firstID;
    const baseURI = `https://student.idtech.com/courses/${courseID}/modules/items/`;
    const outputDir = 'scraped_data';

    // Ensure the output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    let driver: WebDriver;

    try {
        // Set up Firefox options
        const options = new firefox.Options();
        // Uncomment the next line to run Firefox in headless mode
        // options.addArguments("-headless");

        // Initialize the WebDriver
        driver = await new Builder()
            .forBrowser('firefox')
            .setFirefoxOptions(options)
            .build();

        // Set cookies for authentication
        await driver.get('https://student.idtech.com');
        // Add your authentication cookies here
        await driver.manage().addCookie({ name: 'session', value: 'your_session_cookie_here' });
        // Add more cookies as needed

        while (currentID <= lastID) {
            try {
                const url = baseURI + currentID.toString();
                await driver.get(url);

                // Wait for the content to load (adjust timeout as needed)
                await driver.wait(until.elementLocated(By.css('[data-resource-type="wiki_page.body"]')), 10000);

                // Extract text from <span> tag
                const spans = await driver.findElements(By.js('span'));
                const extractedText = await Promise.all(
                    spans.map(async (span) => await span.getText())
                );

                // Save the extracted text to a file
                const fileName = path.join(outputDir, `page_${currentID}.txt`);
                await fs.writeFile(fileName, extractedText.join('\n'));

                console.log(`Scraped and saved content from page ${currentID}`);

                currentID++;
            } catch (error) {
                console.error(`Error processing page ${currentID}:`, error);
                // Continue to the next page even if there's an error
                currentID++;
            }
        }

        if (driver) {
            await driver.quit();
        }
        return true;
    } catch (error) {
        console.error("An error occurred during scraping:", error);
        return error as Error;
    }
};

/**
 * @remarks NLP (Natrual Language Processing) model interface used to start the chatbot
 * @example To train the model on existing or new data
 * ```ts
 * startNLP(1) // -> true
 * ```
 * @example To simply run the model for use in PROD
 * ```ts
 * startNLP(2) // -> true
 * ```
 * @returns bool; true if operation successful, false if any errors arise.
 */
const startNLP = async (operation: 1 | 2): Promise<boolean> => {
    try {
        const pyv = Bun.spawn({
            cmd: ["python3", "--version"],
            stdout: "pipe"
        });
        try {
            const text = await readableStreamToText(pyv.stdout)
            const textarr = text.split(' ')[1].split('.')
            if (text !== '3.6.9' && parseInt(textarr[1]) < 7) {
                console.log(parseInt(textarr[1]))
                throw new Error
            }
        }
        catch {
            console.log("Error: you need python version 3.6.9 or higher to run this program, please install it and try again.")
            return false
        }
    }
    catch {
        console.log("Error: unable to find python version, please make sure you have it installed.")
        return false
    }
    try {
        Bun.spawn({
            cmd: ["python3", `${path.join(__dirname, "model", "main.py")}`, "op"]
        })
    }
    catch { 
        console.log("Unable to find model python file.")
        return false
    }
    return true
}

export {scraper, docServer, startNLP}
docServer().then((Server: Application) => {
    Server.listen(8080);
})