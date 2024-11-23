import React, { useEffect, useRef, useState } from "react";
import OptionsEntry from "./OptionsEntry";
import "./Options.css";

const Options = ({
    setInputText,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
}) => {
    const [textAreas, setTextAreas] = useState([
        {
            id: 1,
            value: `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit
in voluptate velit esse cillum dolore eu
fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in
culpa qui officia deserunt mollit anim
id est laborum.
`,
            readOnly: true,
        },
        { id: 2, value: "This is text area 2", readOnly: true },
        { id: 3, value: "This is text area 3", readOnly: true },
    ]);
    const textAreaRefs = useRef<HTMLTextAreaElement[]>([]);

    const toggleReadOnly = (id: number) => {
        setTextAreas((prevTextAreas) =>
            prevTextAreas.map((textArea) =>
                textArea.id === id
                    ? { ...textArea, readOnly: !textArea.readOnly }
                    : textArea,
            ),
        );
    };

    const changeText = (id: number, newValue: string) => {
        setTextAreas((prevTextAreas) =>
            prevTextAreas.map((textArea) =>
                textArea.id === id
                    ? { ...textArea, value: newValue }
                    : textArea,
            ),
        );
    };

    const rejectChanges = (id: number) => {
        setTextAreas((prevTextAreas) =>
            prevTextAreas.map((textArea) =>
                textArea.id === id
                    ? { ...textArea, value: textArea.value, readOnly: true }
                    : textArea,
            ),
        );
    };

    const handleChange = (id: number, newValue: string) => {
        setTextAreas((prevTextAreas) =>
            prevTextAreas.map((textArea) =>
                textArea.id === id
                    ? { ...textArea, value: newValue }
                    : textArea,
            ),
        );
    };
    const adjustHeight = (textarea: EventTarget) => {
        if (textarea instanceof HTMLTextAreaElement) {
            textarea.style.height = "auto";
            textarea.style.height = `${textarea.scrollHeight}px`;
        }
    };

    useEffect(() => {
        // Adjust the height of all textareas on initial render
        textAreaRefs.current.forEach((textarea) => {
            if (textarea) {
                adjustHeight(textarea);
            }
        });
    }, []);

    useEffect(() => {
        textAreaRefs.current.forEach((textarea, index) => {
            if (textarea) {
                textarea.value = textAreas[index].value;
        }
        });
    }, [textAreas]);

    return (
        <div className="optionsContainer">
            <div className="options">
                <ul>
                    {textAreas.map((textArea, index) => (
                        <li>
                            <OptionsEntry
                                title="Test"
                                toggleReadOnly={toggleReadOnly}
                                textAreaId={textArea.id}
                            >
                                <div key={textArea.id} className="optionsText">
                                    <textarea
                                        ref={(el) =>
                                            el
                                                ? (textAreaRefs.current[index] =
                                                    el)
                                                : null
                                        }
                                        value={textArea.value}
                                        readOnly={textArea.readOnly}
                                        onChange={(e) =>
                                            handleChange(
                                                textArea.id,
                                                e.target.value,
                                            )
                                        }
                                        onInput={(e) => adjustHeight(e.target)}
                                    />
                                </div>
                            </OptionsEntry>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
};
export default Options;
