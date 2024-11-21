import "./Home.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";
import Template from "./Template";
import { useState } from "react";

function Home() {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);

    return (
        <Template>
            <div>
                <div className="centering">
                    <img src="http://127.0.0.1:8000/demo" alt="Cat Image" />
                </div>
                <div className="centering">
                    <img src="http://127.0.0.1:8000/demo" alt="Cat Image" />
                </div>
                <div className="centering">
                    <button className="button">Test</button>
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
