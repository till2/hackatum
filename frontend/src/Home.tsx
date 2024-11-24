import "./Home.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";
import Template from "./Template";
import { useState } from "react";
import Loading from "./components/Loading";
import { API_BASE_URL } from "./config";
import MapsAndOptions from "./components/MapsAndOptions";

import Fullpage, { FullPageSections, FullpageSection } from '@ap.cx/react-fullpage'

function Home() {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const [inputText, setInputText] = useState<string>("");

    const handleTextSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setIsLoading(true);
        // setOutputText("");

        try {
            const response = await fetch(`${API_BASE_URL}/api/transform_text`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify({ text: inputText }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            await new Promise((resolve) =>
                setTimeout(resolve, 1000),
            ); /* Wait for 1 second */

            setInputText(data.output);
        } catch (error) {
            console.error("Error transforming text:", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
            <Fullpage>

<FullPageSections>

            <FullpageSection style={{
                backgroundColor: 'white',
                // padding: '1em',
            }}>
                <div className="container">
                    <div className="content">
                        <h1>Welcome</h1>
                        <p>This is the left section of the page.</p>
                        <div>
                            <form onSubmit={handleTextSubmit}>
                                <textarea
                                    className="input-location"
                                    value={inputText}
                                    onChange={(e) => setInputText(e.target.value)}
                                    placeholder="Enter prompt for AI 🔥"
                                />
                                <button type="submit" className="button">
                                    {isLoading ? <Loading /> : "Submit Text 🚀"}
                                </button>
                            </form>
                        </div>
                    </div>
                    <div className="image"></div>
                </div>
            </FullpageSection>
            <FullpageSection style={{
                backgroundColor: 'lightblue',
                padding: '1em',
            }}>
                <div className="centering">
                     <form onSubmit={handleTextSubmit}>
                         <textarea
                             className="textarea"
                             value={inputText}
                             onChange={(e) => setInputText(e.target.value)}
                             placeholder="Enter prompt for AI 🔥"
                         />
                         <button type="submit" className="button">
                             {isLoading ? <Loading /> : "Submit Text 🚀"}
                         </button>
                     </form>
                 </div>
            </FullpageSection>
            <FullpageSection style={{
                backgroundColor: 'white',
                // padding: '1em',
            }}>
                <div>
                     <MapsAndOptions setInputText={setInputText} />
                 </div>
            </FullpageSection>
            </FullPageSections>

            </Fullpage>
            // <div>
            //     <div className="centering">
            //         <form onSubmit={handleTextSubmit}>
            //             <textarea
            //                 className="textarea"
            //                 value={inputText}
            //                 onChange={(e) => setInputText(e.target.value)}
            //                 placeholder="Enter prompt for AI 🔥"
            //             />
            //             <button type="submit" className="button">
            //                 {isLoading ? <Loading /> : "Submit Text 🚀"}
            //             </button>
            //         </form>
            //     </div>
            //     <div>
            //         <MapsAndOptions setInputText={setInputText} />
            //     </div>
            //     <div className="accordion">
            //         <Accordion
            //             title="How to play?"
            //             content="Upload an image of anything and our AI model will try to extract its most important features! The model analyzes features and characteristics to find good features."
            //             isActive={activeIndex === 0}
            //             index={0}
            //             setActiveIndex={setActiveIndex}
            //         />
            //         <Accordion
            //             title="How is the data processed?"
            //             content="When you upload an image, our computer vision model extracts features and computes an output embedding. We use state-of-the-art deep learning techniques to ensure accurate results while keeping your data private and secure (no OpenAI calls!)."
            //             isActive={activeIndex === 1}
            //             index={1}
            //             setActiveIndex={setActiveIndex}
            //         />
            //         <Accordion
            //             title="When can I try again?"
            //             content="You can upload and match as many images as you'd like (almost)! There's no limit on the number of attempts (almost). Each upload will generate a new output based on the specific features detected in that image."
            //             isActive={activeIndex === 2}
            //             index={2}
            //             setActiveIndex={setActiveIndex}
            //         />
        //             {/* <Accordion
        //                 title="Project Details"
        //                 content="This project was created during HackaTUM 2024, combining computer vision and factor graphs to create a fun interactive project. We used Python with FastAPI for the backend, React for the frontend, and state-of-the-art vision language models for object detection."
        //                 isActive={activeIndex === 3}
        //                 index={3}
        //                 setActiveIndex={setActiveIndex}
        //             />
        //             <Accordion
        //                 title="Who built this?"
        //                 content="We are a team of students from ETH Zürich and HPI who developed this project during HackaTUM 2023. We're passionate about computer vision and wanted to create something fun and interactive that showcases the possibilities of AI technology."
        //                 isActive={activeIndex === 4}
        //                 index={4}
        //                 setActiveIndex={setActiveIndex}
        //             />
        //         </div>
        //     </div>
        // </Template> */}
    );
}

export default Home;
