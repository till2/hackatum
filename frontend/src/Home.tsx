import "./Home.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";
import Template from "./Template";
import { useEffect, useState } from "react";
import Loading from "./components/Loading";
import { API_BASE_URL } from "./config";
import MapsAndOptions from "./components/MapsAndOptions";
import {
    DictLifestyle,
    DictFacts,
    DictEmojis,
    DictHousingFacts,
    DictFactCategories,
} from "./types.ts";

function Home() {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const [inputText, setInputText] = useState<string>("");

    const [lifestyles, setLifestyles] = useState<DictLifestyle>({});

    const [facts, setFacts] = useState<DictFacts>({});

    const [factCategories, setFactCategories] = useState<DictFactCategories>(
        {},
    );

    const [emojis, setEmojis] = useState<DictEmojis>({});

    const [housingFacts, setHousingFacts] = useState<DictHousingFacts>({});

    const handleTextSubmit = async () => {
        setIsLoading(true);
        // setOutputText("");

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/lifeplanner_request`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        Accept: "application/json",
                    },
                    body: JSON.stringify({ text: inputText }),
                },
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            await new Promise((resolve) =>
                setTimeout(resolve, 1000),
            ); /* Wait for 1 second */

            setLifestyles(data.lifestyles);
            setFacts(data.facts);
            setFactCategories(data.fact_categories);
            setEmojis(data.emojis);
            setHousingFacts(data.housing_facts);
        } catch (error) {
            console.error("Error transforming text:", error);
        } finally {
            setIsLoading(false);
        }
    };

    const placeholders = [
        "Find your forever home in just 5 minutes... 🏡",
        "The home you always imagined is just 5 decisions away... 💫",
        "Discover which home best fits to your lifestyle before lunchtime... 🌴",
        "Explore your future home in less than 7 minutes... 🚀",
        "Discover where you can put down your roots in 3 easy steps... 🎋",
        "Your dream home is just 10 clicks away... 🌟",
    ];

    const generateRandomNumber = (min: number, max: number) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    };

    const [placeholderIdx, setPlaceholderIdx] = useState(
        generateRandomNumber(0, placeholders.length - 1),
    );

    useEffect(() => {
        const interval = setInterval(() => {
            setPlaceholderIdx(
                (prevIndex) => (prevIndex + 1) % placeholders.length,
            );
        }, 5000);

        return () => clearInterval(interval);
    }, [placeholders.length]);

    const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (event.shiftKey && event.key === "Enter") {
            event.preventDefault();
            handleTextSubmit();
        }
    };

    return (
        <Template>
            <div>
                <div className="centering">
                    <form
                        onSubmit={(e) => {
                            e.preventDefault();
                            handleTextSubmit();
                        }}
                    >
                        <div className="textareaContainer">
                            <textarea
                                className="textarea"
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                placeholder={placeholders[placeholderIdx]}
                                onKeyDown={handleKeyDown}
                            />
                        </div>
                        <button type="submit" className="button">
                            {isLoading ? <Loading /> : "Find the home to your lifestyle 🚀"}
                        </button>
                    </form>
                </div>
                <div>
                    {Object.keys(lifestyles).length !== 0 ? (
                        <MapsAndOptions
                            setInputText={setInputText}
                            lifestyles={lifestyles}
                            emojis={emojis}
                            factCategories={factCategories}
                            dictFacts={facts}
                            housingFacts={housingFacts}
                        />
                    ) : (
                        <></>
                    )}
                </div>
                <div className="accordion">
                    <Accordion
                        title="How to play?"
                        content="Upload an image of anything and our AI model will try to extract its most important features! The model analyzes features and characteristics to find good features."
                        isActive={activeIndex === 0}
                        index={0}
                        setActiveIndex={setActiveIndex}
                    />
                    <Accordion
                        title="How is the data processed?"
                        content="When you upload an image, our computer vision model extracts features and computes an output embedding. We use state-of-the-art deep learning techniques to ensure accurate results while keeping your data private and secure (no OpenAI calls!)."
                        isActive={activeIndex === 1}
                        index={1}
                        setActiveIndex={setActiveIndex}
                    />
                    <Accordion
                        title="When can I try again?"
                        content="You can upload and match as many images as you'd like (almost)! There's no limit on the number of attempts (almost). Each upload will generate a new output based on the specific features detected in that image."
                        isActive={activeIndex === 2}
                        index={2}
                        setActiveIndex={setActiveIndex}
                    />
                    <Accordion
                        title="Project Details"
                        content="This project was created during HackaTUM 2024, combining computer vision and factor graphs to create a fun interactive project. We used Python with FastAPI for the backend, React for the frontend, and state-of-the-art vision language models for object detection."
                        isActive={activeIndex === 3}
                        index={3}
                        setActiveIndex={setActiveIndex}
                    />
                    <Accordion
                        title="Who built this?"
                        content="We are a team of students from ETH Zürich and HPI who developed this project during HackaTUM 2023. We're passionate about computer vision and wanted to create something fun and interactive that showcases the possibilities of AI technology."
                        isActive={activeIndex === 4}
                        index={4}
                        setActiveIndex={setActiveIndex}
                    />
                </div>
            </div>
        </Template>
    );
}

export default Home;
