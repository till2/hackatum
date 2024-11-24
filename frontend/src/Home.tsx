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
import classNames from "classnames";
import Fullpage, {
    FullPageSections,
    FullpageSection,
} from "@ap.cx/react-fullpage";

import Select from 'react-select'

const options = [
  { value: {lat: 48.132379, lng: 11.576168}, label: 'Munich' },
  { value: {lat: 50.11655996176288, lng: 8.678104088043295}, label: 'Frankfurt' },
  { value: {lat: 52.51258040821299, lng: 13.421978545254927}, label: 'Berlin' }
]

function Home() {
    const [activeIndex, setActiveIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const [inputText, setInputText] = useState<string>("");

    const [lifestyles, setLifestyles] = useState<DictLifestyle>({});

    const [facts, setFacts] = useState<DictFacts>({});

    const [factCategories, setFactCategories] = useState<DictFactCategories>(
        {},
    );

    const [startLocation, setStartLocation] = useState({lat: 48.132379, lng: 11.576168})

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

    // const placeholders = [
    //     "Find your forever home in just 5 minutes... 🏡",
    //     "The home you always imagined is just 5 decisions away... 💫",
    //     "Discover which home best fits to your lifestyle before lunchtime... 🌴",
    //     "Explore your future home in less than 7 minutes... 🚀",
    //     "Discover where you can put down your roots in 3 easy steps... 🎋",
    //     "Your dream home is just 10 clicks away... 🌟",
    // ];
    const placeholders = [
        "Just accepted a position at a tech startup in Amsterdam! I'm searching for a modern apartment with a vibrant community vibe. In my spare time, I love cycling and exploring local cafes.",
        "Planning to move to Vancouver for my master's in environmental science. Looking for a place near the university with easy access to hiking trails and outdoor activities.",
        "As a retired teacher in Seattle, I'm eager to find a cozy home close to the waterfront where I can enjoy sailing and gardening. Seeking a peaceful neighborhood with friendly neighbors.",
        "Recently started my own online business and need a home office space in a lively city like Austin. I enjoy networking events and attending live music shows on weekends.",
        "Moving to Dublin for a research fellowship in biotechnology. Looking for accommodation near the lab with good public transport links. I enjoy reading and weekend getaways.",
        "As a single artist in Barcelona, I'm searching for a loft apartment with plenty of natural light and space for my studio. I love attending art exhibitions and collaborating with other creatives.",
        "Just relocated to Toronto for a senior management role at a finance firm. Seeking a high-rise apartment with excellent city views and amenities. In my free time, I enjoy golfing and fine dining.",
        "Transitioning to life in Melbourne as a freelance graphic designer. Need a stylish studio close to the arts district. I enjoy sketching, visiting museums, and coffee shop hopping.",
        "Moving to Reykjavik to work as a marine biologist. Looking for a home close to the research facilities with stunning natural scenery. I love kayaking and northern lights photography.",
        "As a young professional in New York City, I'm searching for a compact yet comfortable apartment in Manhattan. I thrive on the fast-paced environment and enjoy evening runs in Central Park.",
        "Relocating to Brisbane for a teaching position at a local high school. Seeking a family-friendly neighborhood with good schools and parks. We love barbecues and weekend sports.",
        "Moving to Dubai for a role in international logistics. Looking for a spacious apartment with easy access to the business district. I enjoy desert adventures and fine dining.",
        "As a healthcare worker in Lisbon, I'm searching for affordable housing close to the hospital. In my downtime, I enjoy surfing and exploring historic sites.",
        "Moving to Stockholm to join a sustainable energy company. Seeking a modern flat with eco-friendly features. I love biking, fika with friends, and attending green workshops.",
        "As a recent law graduate in Madrid, I'm looking for a shared apartment near the city center. I enjoy debating, attending court cases, and exploring local tapas bars.",
        "Relocating to Helsinki for a software development position. Need a minimalist apartment with a good workspace. I enjoy gaming, coding side projects, and sauna sessions.",
        "Moving to Zurich to pursue a PhD in physics. Searching for a quiet study environment with access to libraries and research facilities. I enjoy stargazing and hiking.",
        "As a culinary enthusiast in Brussels, I'm looking for a kitchen-friendly apartment near gourmet markets. I love experimenting with new recipes and hosting dinner parties.",
        "Moving to Oslo for a position in urban planning. Seeking a home with modern amenities and proximity to green spaces. I enjoy skiing, reading, and attending theater performances.",
        "As a young parent in Helsinki, I'm searching for a spacious family home with a backyard and access to good schools. We love weekend picnics and outdoor sports activities.",
        "Heading to Singapore for a diplomatic assignment. Looking for a luxurious condo with top-notch security and amenities. I enjoy international cuisine and cultural events.",
        "As a part-time musician in Vienna, I'm seeking a trendy apartment near live music venues and recording studios. I cherish late-night jam sessions and collaborating with fellow artists.",
        "Moving to Auckland to launch my non-profit organization. Need a central location with easy access to community centers and public transport. I enjoy volunteering and outdoor activities.",
        "As a digital marketer in Seoul, I'm searching for a high-tech apartment with smart home features. I love attending tech meetups and exploring the city's nightlife.",
        "Relocating to Cape Town for a wildlife conservation project. Looking for eco-friendly accommodation close to nature reserves. I enjoy hiking, photography, and wildlife watching.",
        "Moving to Montreal to start a career in film production. Seeking a creative loft with space for editing and hosting filming projects. I love indie films and urban exploration.",
        "As a young entrepreneur in Tel Aviv, I'm searching for a modern apartment with a dedicated workspace. I enjoy networking events, startups, and beach outings.",
        "Heading to Miami for a role in international trade. Looking for a vibrant neighborhood with easy beach access. I enjoy sailing, salsa dancing, and nightlife.",
        "As a freelance translator in Prague, I'm seeking a quiet apartment near libraries and cafes. I love languages, reading, and exploring historical sites.",
        "Moving to Nairobi for a position with an NGO focused on education. Searching for safe, community-oriented housing close to schools and parks. I enjoy mentoring and community service.",
        "As a 26-year-old yoga instructor in Bangkok, I'm looking for a serene apartment near yoga studios and wellness centers. I enjoy meditation, healthy cooking, and beach retreats.",
        "Relocating to Helsinki for a master's program in economics. Seeking student-friendly housing with study areas and social spaces. I enjoy budgeting workshops and local markets.",
        "Moving to Lisbon to work as a marine engineer. Looking for a waterfront apartment with stunning ocean views. I enjoy sailing, scuba diving, and seaside jogging.",
        "As a 39-year-old journalist in Chicago, I'm searching for a central apartment close to media houses and cultural hotspots. I enjoy investigative reporting and urban photography.",
        "Heading to Dublin for a fellowship in renewable energy. Seeking a green-certified home with access to cycling paths and parks. I enjoy sustainable living and outdoor festivals.",
        "As a fashion designer in Milan, I'm looking for a chic loft with ample space for design and showcases. I love attending fashion shows and trend workshops.",
        "Moving to Boston for a research position in neuroscience. Seeking a residence near the university with a quiet study environment. I enjoy reading scientific journals and running.",
        "As a 23-year-old dancer in Rio de Janeiro, I'm searching for an apartment close to dance studios and vibrant nightlife. I love samba, fitness, and cultural festivals.",
        "Relocating to Amsterdam for a position in international relations. Looking for a canal-side flat with historic charm and modern amenities. I enjoy cycling, museums, and coffee shops.",
        "Moving to Brussels to work with the European Union. Seeking a centrally located apartment with easy access to government buildings. I enjoy multilingual conversations and fine dining.",
        "As a 34-year-old photographer in San Francisco, I'm searching for a home with plenty of natural light and scenic views. I love capturing cityscapes and exploring new neighborhoods.",
        "Heading to Melbourne for a PhD in marine biology. Looking for a research-friendly apartment near the coast with access to marine reserves. I enjoy diving, kayaking, and marine conservation.",
        "As a 28-year-old app developer in Bangalore, I'm seeking a tech-friendly apartment with high-speed internet and smart home devices. I enjoy coding hackathons and attending tech conferences.",
        "Moving to Toronto to join a leading architectural firm. Looking for a modern condo with sleek designs and proximity to design studios. I enjoy sketching, traveling, and attending art exhibitions.",
        "As a 31-year-old social media influencer in Los Angeles, I'm searching for a stylish apartment with photogenic spots and good lighting. I love content creation, fashion, and attending events.",
        "Relocating to Edinburgh for a teaching role at a prestigious university. Seeking a historic home with character and proximity to academic resources. I enjoy literature, hiking, and local history.",
        "Moving to Sydney to work in marine conservation. Looking for an eco-friendly home near the harbor with outdoor space. I enjoy snorkeling, beach volleyball, and environmental activism.",
        "As a 27-year-old software tester in Berlin, I'm searching for a minimalist apartment with a dedicated workspace and close to tech hubs. I enjoy gaming, coffee hopping, and attending meetups.",
        "Heading to Vancouver for a graphic design internship. Seeking a trendy apartment with artistic flair and access to design communities. I love illustration, street art, and indie films.",
        "As a 36-year-old project manager in Dubai, I'm looking for a spacious villa with modern amenities and a private garden. I enjoy hosting gatherings, gardening, and desert safaris.",
        "Moving to Munich to join a leading automotive company as a mechanical engineer. I'm seeking a modern apartment in the Schwabing district with easy access to public transport and parks. In my free time, I enjoy exploring Bavarian castles, cycling along the Isar River, and savoring local cuisine at traditional beer gardens.",
    ];

    const generateRandomNumber = (min: number, max: number) => {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    };

    const [placeholder, setPlaceholder] = useState(
        placeholders[generateRandomNumber(0, placeholders.length - 1)],
    );

    const [fadeClass, setFadeClass] = useState<string>("fade-in");

    useEffect(() => {
        const interval = setInterval(() => {
            setFadeClass("fade-out");
            setTimeout(() => {
                setPlaceholder((prev) => {
                    const currentIndex = placeholders.indexOf(prev);
                    const nextIndex = (currentIndex + 1) % placeholders.length;
                    return placeholders[nextIndex];
                });
                setFadeClass("fade-in");
            }, 500);
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
        <Fullpage>
            <FullPageSections>
                <FullpageSection>
                <Template>
                    <div style={{width: "50%", margin: "auto"}}>
                        <Select defaultValue={options[0]} options={options} styles={{
                                    control: (baseStyles, state) => ({
                                        ...baseStyles,
                                        borderColor: state.isFocused ? 'grey' : 'orange',
                                        // width: '50%',
                                        height: '80px',
                                        // justifyItems: 'center',
                                        // margin: "auto",
                                    }),
                                    menu: (provided) => ({
                                        ...provided,
                                        // background: 'transparent',
                                        // width: '50%',
                                        // margin: "auto",
                                        // display: "flex",
                                        // // justifyContent: "center"
                                        // alignitems: 'center'
                                    }),
                                }}
                                onChange={(choice) => setStartLocation(choice.value)}/>


                    </div>
                </Template>
                </FullpageSection>
                <FullpageSection>
                    <Template>
                        <>
                            <div className="centering">
                                <form
                                    onSubmit={(e) => {
                                        e.preventDefault();
                                        handleTextSubmit();
                                    }}
                                >
                                    <div className="textareaContainer">
                                        <textarea
                                            className={classNames(
                                                "textarea",
                                                fadeClass,
                                            )}
                                            value={inputText}
                                            onChange={(e) =>
                                                setInputText(e.target.value)
                                            }
                                            placeholder={placeholder}
                                            onKeyDown={handleKeyDown}
                                        />
                                    </div>
                                    <button type="submit" className="button">
                                        {isLoading ? (
                                            <Loading />
                                        ) : (
                                            "Find the home to your lifestyle 🚀"
                                        )}
                                    </button>
                                </form>
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

                        </>
                    </Template>
                </FullpageSection>
                {Object.keys(lifestyles).length !== 0 ? (
                    <FullpageSection>
                        <Template>
                            <div>
                                <form
                                    onSubmit={(e) => {
                                        e.preventDefault();
                                        handleTextSubmit();
                                    }}
                                >
                                    <div
                                        style={{
                                            display: "flex",
                                            flexDirection: "row",
                                            justifyContent: "center",
                                            height: "10vh",
                                            width: "100%",
                                            marginBottom: "1rem",
                                        }}
                                    >
                                        <div className="textareaContainer">
                                            <textarea
                                                className={classNames(
                                                    "textarea",
                                                    fadeClass,
                                                )}
                                                style={{ width: "66vw" }}
                                                value={inputText}
                                                onChange={(e) =>
                                                    setInputText(e.target.value)
                                                }
                                                placeholder={placeholder}
                                                onKeyDown={handleKeyDown}
                                            />
                                        </div>
                                        <button
                                            type="submit"
                                            className="button"
                                            style={{
                                                width: "10vw",
                                                height: "70%",
                                                justifySelf: "center",
                                            }}
                                        >
                                            {isLoading ? (
                                                <Loading />
                                            ) : (
                                                "Find the home to your lifestyle 🚀"
                                            )}
                                        </button>
                                    </div>
                                </form>
                                <MapsAndOptions
                                    setInputText={setInputText}
                                    lifestyles={lifestyles}
                                    emojis={emojis}
                                    factCategories={factCategories}
                                    dictFacts={facts}
                                    housingFacts={housingFacts}
                                    startLocation={startLocation}
                                />
                            </div>
                        </Template>
                    </FullpageSection>
                ) : (
                    <></>
                )}
            </FullPageSections>
        </Fullpage>
    );
}

export default Home;
