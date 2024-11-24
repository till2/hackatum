import React, { useEffect, useRef, useState } from "react";
import OptionsEntry from "./OptionsEntry";
import "./Options.css";
import {
    DictLifestyle,
    DictEmojis,
    DictFactCategories,
    DictFacts,
    DictHousingFacts,
} from "../types.ts";

type TextAreas = {
    id: number;
    value: string;
    readOnly: boolean;
    emojis: string;
}[];

const Options = ({
    setInputText,
    lifestyles,
    emojis,
    factCategories,
    dictFacts,
    housingFacts,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
    lifestyles: DictLifestyle;
    emojis: DictEmojis;
    factCategories: DictFactCategories;
    dictFacts: DictFacts;
    housingFacts: DictHousingFacts;
}) => {
    const [textAreas, setTextAreas] = useState<TextAreas>([]);

    useEffect(() => {
        const textAreasStart: TextAreas = [];
        if (Object.keys(lifestyles).length === 0) {
            return;
        }
        let y = 1;
        Object.entries(lifestyles).forEach(([key, value]) => {
            let newValue = "";
            const numberEntries = Object.keys(value).length;
            let i = 0;
            let emojisArray = Object.values(emojis[y]);

            Object.values(value).forEach((value) => {
                if (i === numberEntries - 1) {
                    newValue += value;
                    return;
                }
                newValue += `${value}, `;
                i++;
            });

            textAreasStart.push({
                id: parseInt(key),
                value: newValue,
                readOnly: true,
                emojis: emojisArray.join(""),
            });
            y++
        });
        setTextAreas(textAreasStart);
    }, [lifestyles, emojis]);

    const textAreaRefs = useRef<HTMLTextAreaElement[]>([]);
    const [oldText, setOldText] = useState<string>("");

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

        textAreaRefs.current.forEach((textarea) => {
            if (textarea) {
                adjustHeight(textarea);
            }
        });
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
    }, [textAreas]);

    return (
        <div className="optionsContainer">
            <div className="options">
                <ul>
                    {textAreas.map((textArea, index) => (
                        <li key={index}>
                            <OptionsEntry
                                title={textArea.emojis}
                                toggleReadOnly={toggleReadOnly}
                                textAreaId={textArea.id}
                                text={textArea.value}
                                oldText={oldText}
                                setOldText={setOldText}
                                changeText={changeText}
                                setInputText={setInputText}
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
