import React, { useState } from "react";
import "./Options.css";

interface OptionsEntryProps {
    title: string;
    children: React.ReactElement;
    toggleReadOnly: (id: number) => void;
    textAreaId: number;
    oldText: string;
    changeText: (id: number, newValue: string) => void;
    setOldText: (newValue: string) => void;
    text: string;
    setInputText: React.Dispatch<React.SetStateAction<string>>;
}

const OptionsEntry: React.FC<OptionsEntryProps> = ({
    title,
    children,
    toggleReadOnly,
    oldText,
    textAreaId,
    changeText,
    setOldText,
    text,
    setInputText,
}) => {
    const [readOnly, setReadOnly] = useState(true);
    return (
        <div className="optionsEntry">
            <div className="optionsHeader">
                <h2 className="optionsTitle">{title}</h2>
                {readOnly ? (
                    <div>
                        <button
                            onClick={() => {
                                toggleReadOnly(textAreaId);
                                setReadOnly(!readOnly);
                                setOldText(text);
                            }}
                        >
                            Modify
                        </button>
                        <button onClick={() => setInputText(text)}>
                            Apply
                        </button>
                    </div>
                ) : (
                    <div>
                        <button
                            onClick={() => {
                                setReadOnly(!readOnly);
                                toggleReadOnly(textAreaId);
                                changeText(textAreaId, oldText);
                            }}
                        >
                            Reject
                        </button>
                        <button
                            onClick={() => {
                                setReadOnly(!readOnly);
                                toggleReadOnly(textAreaId);
                            }}
                        >
                            Accept
                        </button>
                    </div>
                )}
            </div>
            <div className="optionsContent">{children}</div>
        </div>
    );
};

export default OptionsEntry;
