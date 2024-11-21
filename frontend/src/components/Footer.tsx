import Logo from "./Logo";
import ETHLogo from "../../assets/logo-eth-white-1.png";
import HPILogo from "../../assets/hpi_logo_white.png";
import "./Footer.css";

const Footer = () => {
    return (
        <div className="footer">
            <a href="https://ethz.ch/en.html" target="_blank" rel="noopener noreferrer">
                <Logo src={ETHLogo} width="200px" />
            </a>
            <a href="https://hpi.de/en/" target="_blank" rel="noopener noreferrer">
                <Logo src={HPILogo} width="100px" />
            </a>
        </div>
    );
};
export default Footer;
