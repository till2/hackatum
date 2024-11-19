import "./BlogEntry.css";

const BlogEntry = ({
    title,
    children,
}: {
    title: string;
    children: JSX.Element[];
}) => {
    return (
        <div className="blogEntry">
            <h2 className="blogTitle">{title}</h2>
            <div className="blogContent">{children}</div>
        </div>
    );
};

export default BlogEntry;
