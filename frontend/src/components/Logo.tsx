const Logo = ({ src, width }: { src: string; width: string }) => {
    return (
        <div style={{ justifySelf: "center" }}>
            <img width={width} src={src} style={{marginRight: "40px", marginLeft: "40px"}}/>
        </div>
    );
};

export default Logo;
