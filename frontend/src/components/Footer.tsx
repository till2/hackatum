import React from "react";
import Logo from "./Logo";
import ETHLogo from "../../assets/logo-eth-white-1.png";
import HPILogo from "../../assets/hpi_logo_white.png";
import "./Footer.css";

const Footer: React.FC = () => {
    return (
        <footer className="footer">
            <a
                href="https://ethz.ch/en.html"
                target="_blank"
                rel="noopener noreferrer"
                className="footer-link"
            >
                <Logo src={ETHLogo} width="200px" />
            </a>
            <a
                href="https://hpi.de/en/"
                target="_blank"
                rel="noopener noreferrer"
                className="footer-link"
            >
                <Logo src={HPILogo} width="100px" />
            </a>
        </footer>
    );
};

export default Footer;
