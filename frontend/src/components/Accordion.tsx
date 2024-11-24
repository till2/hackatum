import { Dispatch, SetStateAction } from "react";

interface AccordionProps {
  title: string;
  content: string;
  isActive: boolean;
  index: number;
  setActiveIndex: Dispatch<SetStateAction<number | null>>;
}

const Accordion = ({ title, content, isActive, index, setActiveIndex }: AccordionProps) => {
    return (
        <div className="accordion-item">
            <div
                className="accordion-header"
                onClick={() => setActiveIndex(isActive ? null : index)}
            >
                <div className="accordion-title">{title}</div>
                <div className="accordion-status">{isActive ? "-" : "+"}</div>
            </div>
            <div className={`accordion-content ${isActive ? 'active' : ''}`} style={{ textAlign: 'left', marginLeft: '5px', backgroundColor: 'var(--first-color)' }}>
                {content}
            </div>
        </div>
    );
};

export default Accordion;
