import React from "react";
import "./Options.css";

interface OptionsEntryProps {
    title: string;
    children: React.ReactElement;
    setInputText: React.Dispatch<React.SetStateAction<string>>;
}

const OptionsEntry: React.FC<OptionsEntryProps> = ({
    title,
    setInputText,
    children,
}) => {
    const handleClick = () => {
        setInputText(children.props.children.props.children.props.children);
    };
    return (
        <button className="optionsEntry" onClick={handleClick}>
            <h2 className="optionsTitle">{title}</h2>
            <div className="optionsContent">{children}</div>
        </button>
    );
};

export default OptionsEntry;
