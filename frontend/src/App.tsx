import "./App.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";
import Footer from "./components/Footer";

function App() {
    return (
        <>
            <div className="app">
                <div>
                    <div
                        style={{
                            display: "flex",
                            width: "100vw",
                            flexDirection: "column",
                            alignItems: "center",
                        }}
                    >
                        <img src="http://127.0.0.1:8000/demo" alt="Cat Image" />
                    </div>
                    <div
                        style={{
                            display: "flex",
                            width: "100vw",
                            flexDirection: "column",
                            alignItems: "center",
                        }}
                    >
                        <button className="button">Test</button>
                    </div>
                    <div className="accordion">
                        <Accordion title="test" content="test" />
                    </div>
                </div>
                <Footer />
            </div>
        </>
    );
}

export default App;
