use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Field, Fields, Ident, ItemStruct, Type, Visibility};

#[proc_macro_attribute]
pub fn layer(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse o input para uma estrutura ItemStruct
    let mut input = parse_macro_input!(item as ItemStruct);
    let struct_name = &input.ident;

    // Certifique-se de que os campos são nomeados
    let fields = if let Fields::Named(ref mut fields) = input.fields {
        fields
    } else {
        return syn::Error::new_spanned(input, "Expected a struct with named fields")
            .to_compile_error()
            .into();
    };

    // Definir os novos campos a serem adicionados
    let additional_fields = vec![
        ("last_input", "Tensor"),
        ("last_output", "Tensor"),
        ("input_col_size", "usize"),
        ("input_row_size", "usize"),
        ("output_col_size", "usize"),
        ("output_row_size", "usize"),
    ];

    // Adicionar os novos campos
    for (field_name, field_type) in additional_fields {
        let field = Field {
            attrs: vec![],
            vis: Visibility::Public(syn::token::Pub::default()),
            ident: Some(Ident::new(field_name, proc_macro2::Span::call_site())),
            colon_token: Some(Default::default()),
            ty: syn::parse_str::<Type>(field_type).expect("Failed to parse type"),
            mutability: syn::FieldMutability::None,
        };
        fields.named.push(field);
    }

    // Gerar a nova definição da struct com os campos adicionados
    let expanded_struct = quote! {
        #input
    };

    // Gerar a implementação parcial da trait Layer
    let expanded_impl = quote! {
        impl LayerDefaultTrait for #struct_name {
            fn get_output_shape(&self) -> (usize, usize) {
                (self.output_row_size, self.output_col_size)
            }

            fn get_input_shape(&self) -> (usize, usize) {
                (self.input_row_size, self.input_col_size)
            }
        }
    };

    // Juntar a definição da struct e a implementação da trait
    let expanded = quote! {
        #expanded_struct

        #expanded_impl
    };

    TokenStream::from(expanded)
}