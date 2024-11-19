import "./Home.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";
import Template from "./Template";

function Home() {
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
                    <Accordion title="Test" content="test" />
                </div>
            </div>
        </Template>
    );
}

export default Home;
