import "./Blog.css";
import "./components/Accordion.css";
import Template from "./Template";
import BlogEntry from "./components/BlogEntry";

function Blog() {
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
                                        alt="Merle (Dog)"
                                        style={{ width: "80px", height: "auto", padding: "25px" }}
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
                            <BlogEntry title="First Stop: Marienplatz">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/marienplatz.jpg"
                                        alt="Marienplatz Munich"
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
                            <BlogEntry title="Breakfast Time">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/breakfast.jpg"
                                        alt="Team Breakfast"
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
                            <BlogEntry title="Made it to the Hackathon!">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/arrival.jpg"
                                        alt="Hackathon Arrival"
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
                            <BlogEntry title="Opening Ceremony">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/opening_ceremony.jpg"
                                        alt="Opening Ceremony"
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
                            <BlogEntry title="Coding Night">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/hacking.jpg"
                                        alt="Team Hacking"
                                    />
                                    <div className="blogText">
                                        <p>
                                            4 hours in and already deep into CSS formatting ☕️
                                            We're setting up a basic React frontend and start exploring langgraph. 
                                            Also debugging some buttons for an hour though - send help. 
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="Power Nap">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/blog/sleep.jpg"
                                        alt="Taking a Break"
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
                    </ul>
                </div>
            </div>
        </Template>
    );
}

export default Blog;
