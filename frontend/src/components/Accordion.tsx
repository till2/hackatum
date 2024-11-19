import { useState } from "react";

const Accordion = ({ title, content }: { title: string; content: string }) => {
    const [isActive, setIsActive] = useState(false);

    return (
        <div className="accordion-item">
            <div
                className="accordion-header"
                onClick={() => setIsActive(!isActive)}
            >
                <div className="accordion-title">{title}</div>
                <div className="accordion-status">{isActive ? "-" : "+"}</div>
            </div>
            {isActive && <div className="accordion-content">{content}</div>}
        </div>
    );
};

export default Accordion;
