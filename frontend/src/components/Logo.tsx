import "./Logo.css";

const Logo = ({ src, width }: { src: string; width: string }) => {
    return (
        <div className="logoContainer">
            <img width={width} src={src} className="logo" />
        </div>
    );
};

export default Logo;
