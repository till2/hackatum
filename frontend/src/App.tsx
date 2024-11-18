import { useState } from "react";
import "./App.css";
import "./components/Accordion.css";
import Accordion from "./components/Accordion";

function App() {
    return (
        <>
            <div>
                <img src="http://127.0.0.1:8000/demo" alt="Cat Image" />
            </div>
            <button>Test</button>
            <div className="Accordion">
                <Accordion title="test" content="test" />
            </div>
        </>
    );
}

export default App;
