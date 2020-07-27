use actix_files::Files;
use actix_web::{middleware, web, App, HttpServer, Responder};
use rust_bert::pipelines::generation::{GPT2Generator, GenerateConfig, LanguageGenerator};
use serde::Deserialize;

#[derive(Deserialize)]
struct Data {
    context: String,
}

async fn generate(data: web::Json<Data>) -> impl Responder {
    predict(&data.context)
}

#[actix_rt::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=debug");
    env_logger::init();

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/generate", web::post().to(generate))
            .service(Files::new("/", "static/").index_file("index.html"))
    })
    .bind("127.0.0.1:8000")?
    .run()
    .await
}

fn predict(text: &str) -> String {
    //    Set-up masked LM model
    let generate_config = GenerateConfig {
        max_length: 30,
        do_sample: true,
        num_beams: 5,
        temperature: 1.1,
        num_return_sequences: 1,
        ..Default::default()
    };
    let model = GPT2Generator::new(generate_config).unwrap();

    model.generate(Some(vec![text]), None)[0].to_string()
}
