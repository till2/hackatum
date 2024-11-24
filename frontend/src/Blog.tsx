import React, { useState, MouseEvent, useRef, useEffect } from 'react';
import './Blog.css';
import './components/Accordion.css';
import Template from './Template';
import BlogEntry from './components/BlogEntry';

const Blog: React.FC = () => {
    const [isHovered, setIsHovered] = useState<boolean>(false);
    const [isRotating, setIsRotating] = useState<boolean>(false);
    const imgRef = useRef<HTMLImageElement>(null);

    const handleMouseEnter = (event: MouseEvent<HTMLImageElement>) => {
        setIsHovered(true);
        if (!isRotating) {
            setIsRotating(true);
        }
    };

    const handleMouseLeave = (event: MouseEvent<HTMLImageElement>) => {
        setIsHovered(false);
    };

    const handleAnimationEnd = () => {
        if (!isHovered) {
            setIsRotating(false);
        }
    };

    useEffect(() => {
        const imgElement = imgRef.current;
        if (imgElement) {
            imgElement.addEventListener('animationend', handleAnimationEnd);
        }

        return () => {
            if (imgElement) {
                imgElement.removeEventListener('animationend', handleAnimationEnd);
            }
        };
    }, [isHovered]);

    return (
        <Template>
            <div className="blogContainer">
                <div className="blog">
                    <h1>Our HackaTUM Journey</h1>
                    <ul>
                        <li>
                            <BlogEntry title="Welcome to our Dev-Blog!">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/dog.png"
                                        alt="Dog"
                                        ref={imgRef}
                                        className={`dog-blog-image ${isRotating ? 'continuous-rotate' : ''}`}
                                        style={{ width: "80px", height: "auto", padding: "25px", borderRadius: "8px" }}
                                        onMouseEnter={handleMouseEnter}
                                        onMouseLeave={handleMouseLeave}
                                    />
                                    <div className="blogText">
                                        <p style={{ paddingTop: "20px" }}>
                                            Hey there! <br/> <br/>
                                            We're Artiom, Constantin and Till - a team of three master's students from ETH Zurich and Hasso Plattner Institute, specializing in 
                                            Mathematics, Machine Learning and Data Engineering. Looking at the weekend ahead, 
                                            we're excited about a fun and productive time :) <br/><br/>
                                            {/* (We're not very keen on once again becoming front-end devs for a weekend though.. 😅) */}
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Fri @ 11:00] - Breakfast Time">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/breakfast.jpg"
                                        alt="Team Breakfast"
                                        style={{ minWidth: "200px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            Constantin recommended this café near our place ☕️ - Nothing beats planning a hackathon 
                                            project over fresh coffee and warm croissants. Also, we're definitely gonna need that energy later! 
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Fri @ 13:00] - Next Stop: Marienplatz">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/marienplatz.jpg"
                                        alt="Marienplatz Munich"
                                        style={{ minWidth: "150px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            Made it to Munich! We arrived by train and went to check out Marienplatz first 🏰 Found out there's an elevator 
                                            up the Frauenkirche - a pleasant suprise 🆙 <br/> Got some sweet views of the city before 
                                            heading to our sleeping place for the night. Ready for the weekend!
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Fri @ 16:30] - Made it to the Hackathon!">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/arrival.jpg"
                                        alt="Hackathon Arrival"
                                        style={{ width: "250px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            The venue is packed! Everyone's talking about their ideas and the challenges they want 
                                            to tackle.
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Fri @ 17:00] - Opening Ceremony">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/opening_ceremony.jpg"
                                        alt="Opening Ceremony"
                                        style={{ minWidth: "200px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            Opening ceremony had a radio presenter. 🎤 We brainstormed some ideas for 
                                            potential projects, though none of the challenges really fit building and training our own 
                                            AI model. Still, we came up with some good stuff and could also already cross out many of the challenges.
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Sat @ 03:00] - Coding Night">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/hacking.jpg"
                                        alt="Team Hacking"
                                        style={{ minWidth: "200px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            A couple of hours in and already deep into CSS formatting ☕️
                                            We're setting up a basic React frontend and start exploring langgraph. 
                                            Also debugging some buttons for an hour though - send help. 
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Sat @ 05:00] - Power Nap">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/sleep.jpg"
                                        alt="Taking a Break"
                                        style={{ minWidth: "200px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            5 AM and starting to get really tired. Our strategy is to sleep the first day, so that we're 
                                            not sleep deprived on saturday. Sunday we'll manage. To sleep, we brought some sleeping mats and bags. 
                                            So now we're taking a quick power nap and then back to coding.
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="[Sat @ 20:00] - Concert Break">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/concert.jpg"
                                        alt="Concert Break"
                                        style={{ minWidth: "200px", borderRadius: "8px" }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            Taking a well-deserved break :) We bought tickets to a concert by ENNIO happening nearby 
                                            and decided to take 4 hours off from coding to enjoy some epic bass. We basically spent 
                                            our sleeping time budget for today on the concert, but it was well worth it!
                                            Sometimes the best ideas come when you step away from the keyboard for a bit and drink some beer.
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                    </ul>
                </div>
            </div>
        </Template>
    );
};

export default Blog;
