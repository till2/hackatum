import Logo from "./Logo";
import ETHLogo from "../assets/logo-eth-white-1.png";
import HPILogo from "../assets/hpi_logo_white.png";
import "./Footer.css";

const Footer = () => {
    return (
        <div className="footer">
            <Logo src={ETHLogo} width="200px" />
            <Logo src={HPILogo} width="100px" />
        </div>
    );
};
export default Footer;
