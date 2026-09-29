CREATE EXTENSION IF NOT EXISTS vector;




CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,

    wine_name TEXT NOT NULL,
    category TEXT,
    color TEXT,
    region TEXT,
    grape_variety TEXT,
    description TEXT,
    winery TEXT,
    slug TEXT NOT NULL UNIQUE,
    photo_name TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



CREATE TABLE IF NOT EXISTS product_embeddings (
    id BIGSERIAL PRIMARY KEY,

    product_id BIGINT NOT NULL
        REFERENCES products(id)
        ON DELETE CASCADE,

    filename TEXT NOT NULL,

    image_type TEXT NOT NULL
        CHECK (image_type IN ('full', 'crop')),

    model_version TEXT NOT NULL DEFAULT 'finetuned',

    embedding VECTOR(768) NOT NULL,

    UNIQUE (
        product_id,
        image_type,
        model_version
    )
);



CREATE INDEX IF NOT EXISTS product_embeddings_cosine_idx
    ON product_embeddings
    USING hnsw (embedding vector_cosine_ops);


CREATE INDEX IF NOT EXISTS product_embeddings_lookup_idx
    ON product_embeddings (
        model_version,
        image_type
    );

CREATE INDEX IF NOT EXISTS product_embeddings_product_id_idx
    ON product_embeddings (product_id);


DROP INDEX IF EXISTS products_wine_name_uidx;


CREATE TABLE IF NOT EXISTS bottle_embeddings (
    id BIGSERIAL PRIMARY KEY,
    classname TEXT NOT NULL,
    filename TEXT NOT NULL DEFAULT '',
    embedding_mode TEXT NOT NULL CHECK (embedding_mode IN ('label', 'full')),
    model_version TEXT NOT NULL DEFAULT 'finetuned',
    embedding VECTOR(768) NOT NULL,
    UNIQUE (classname, filename, embedding_mode, model_version)
);


ALTER TABLE bottle_embeddings
    ALTER COLUMN embedding TYPE VECTOR(768)
    USING embedding::vector(768);


CREATE INDEX IF NOT EXISTS bottle_embeddings_cosine_idx
    ON bottle_embeddings USING hnsw (embedding vector_cosine_ops);


CREATE INDEX IF NOT EXISTS bottle_embeddings_lookup_idx
    ON bottle_embeddings (model_version, embedding_mode);
