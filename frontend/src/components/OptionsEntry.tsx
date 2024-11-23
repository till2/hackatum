import React, { useState } from "react";
import "./Options.css";

interface OptionsEntryProps {
    title: string;
    children: React.ReactElement;
    toggleReadOnly: (id: number) => void;
    textAreaId: number;
    rejectChanges: (id: number) => void;
}

const OptionsEntry: React.FC<OptionsEntryProps> = ({
    title,
    children,
    toggleReadOnly,
    textAreaId,
    rejectChanges
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
                            }}
                        >
                            Modify
                        </button>
                        <button>Apply</button>
                    </div>
                ) : (
                    <div>
                        <button onClick={() => rejectChanges(textAreaId)}>Reject</button>
                        <button>Accept</button>
                    </div>
                )}
            </div>
            <div className="optionsContent">{children}</div>
        </div>
    );
};

export default OptionsEntry;
