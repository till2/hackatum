import Logo from "./Logo";
import ETHLogo from "../../assets/logo-eth-white-1.png";
import "./Header.css";
import { Link, useLocation } from "react-router-dom";

const Header = () => {
    const location = useLocation();
    const link = location.pathname === "/blog" ? "/" : "/blog";
    const linkText = location.pathname === "/blog" ? "Home" : "Blog";
    return (
        <div className="header">
            <Link to="/">
                <Logo src={ETHLogo} width="200px" />
            </Link>
            <Link style={{ margin: "20px" }} to={link}>{linkText}</Link>
        </div>
    );
};
export default Header;
