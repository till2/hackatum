import React from "react";
import OptionsEntry from "./OptionsEntry";
import "./Options.css";

const Options = ({setInputText} : {setInputText: React.Dispatch<React.SetStateAction<string>>}) => {
    return (
        <div className="optionsContainer">
            <div className="options">
                <ul>
                    <li>
                        <OptionsEntry title="Test" setInputText={setInputText}>
                            <div className="optionsEntryContent">
                                <div className="optionsText">
                                    <p>
                                        Lorem ipsum dolor sit amet, consectetur
                                        adipiscing elit, sed do eiusmod tempor
                                        incididunt ut labore et dolore magna
                                        aliqua. Ut enim ad minim veniam, quis
                                        nostrud exercitation ullamco laboris
                                        nisi ut aliquip ex ea commodo consequat.
                                        Duis aute irure dolor in reprehenderit
                                        in voluptate velit esse cillum dolore eu
                                        fugiat nulla pariatur. Excepteur sint
                                        occaecat cupidatat non proident, sunt in
                                        culpa qui officia deserunt mollit anim
                                        id est laborum.
                                    </p>
                                </div>
                            </div>
                        </OptionsEntry>
                    </li>
                    <li>
                        <OptionsEntry title="Test" setInputText={setInputText}>
                            <div className="optionsEntryContent">
                                <div className="optionsText">
                                    <p>
                                        Lorem ipsum dolor sit amet, consectetur
                                        adipiscing elit, sed do eiusmod tempor
                                        incididunt ut labore et dolore magna
                                        aliqua. Ut enim ad minim veniam, quis
                                        nostrud exercitation ullamco laboris
                                        nisi ut aliquip ex ea commodo consequat.
                                        Duis aute irure dolor in reprehenderit
                                        in voluptate velit esse cillum dolore eu
                                        fugiat nulla pariatur. Excepteur sint
                                        occaecat cupidatat non proident, sunt in
                                        culpa qui officia deserunt mollit anim
                                        id est laborum.
                                    </p>
                                </div>
                            </div>
                        </OptionsEntry>
                    </li>
                    <li>
                        <OptionsEntry title="Test" setInputText={setInputText}>
                            <div className="optionsEntryContent">
                                <div className="optionsText">
                                    <p>
                                        Lorem ipsum dolor sit amet, consectetur
                                        adipiscing elit, sed do eiusmod tempor
                                        incididunt ut labore et dolore magna
                                        aliqua. Ut enim ad minim veniam, quis
                                        nostrud exercitation ullamco laboris
                                        nisi ut aliquip ex ea commodo consequat.
                                        Duis aute irure dolor in reprehenderit
                                        in voluptate velit esse cillum dolore eu
                                        fugiat nulla pariatur. Excepteur sint
                                        occaecat cupidatat non proident, sunt in
                                        culpa qui officia deserunt mollit anim
                                        id est laborum.
                                    </p>
                                </div>
                            </div>
                        </OptionsEntry>
                    </li>
                </ul>
            </div>
        </div>
    );
};
export default Options;
