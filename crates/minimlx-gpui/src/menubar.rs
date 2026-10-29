#![allow(unexpected_cfgs)]

use cocoa::{
    appkit::{
        NSButton, NSControl, NSImage, NSMenu, NSMenuItem, NSStatusBar, NSStatusItem,
        NSVariableStatusItemLength,
    },
    base::{NO, YES, id, nil},
    foundation::{NSAutoreleasePool, NSRange, NSString},
};
use objc::{
    class,
    declare::ClassDecl,
    msg_send,
    rc::StrongPtr,
    runtime::{Class, Object, Sel},
    sel, sel_impl,
};
use std::{
    cell::RefCell,
    sync::{OnceLock, mpsc},
};

const STATUS_ICON: &[u8] = include_bytes!("../assets/brand/MenuBarIcon.png");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    ShowWindow,
    HideWindow,
    StartService,
    StopService,
    Quit,
}

static MENU_SENDER: OnceLock<mpsc::Sender<MenuAction>> = OnceLock::new();
static TARGET_CLASS: OnceLock<usize> = OnceLock::new();

struct NativeMenu {
    _status_item: StrongPtr,
    _menu: StrongPtr,
    _target: StrongPtr,
    service_item: StrongPtr,
    model_item: StrongPtr,
    window_action_item: StrongPtr,
    service_action_item: StrongPtr,
}

thread_local! {
    static NATIVE_MENU: RefCell<Option<NativeMenu>> = const { RefCell::new(None) };
}

pub fn install(labels: MenuLabels, running: bool) -> mpsc::Receiver<MenuAction> {
    let (sender, receiver) = mpsc::channel();
    let _ = MENU_SENDER.set(sender);
    unsafe { install_native_menu(labels, running) };
    receiver
}

#[derive(Clone)]
pub struct MenuLabels {
    pub app_name: String,
    pub service_status: String,
    pub service_emphasis: StatusEmphasis,
    pub model_status: String,
    pub show_window: String,
    pub hide_window: String,
    pub start_service: String,
    pub stop_service: String,
    pub quit: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusEmphasis {
    Neutral,
    Running,
    InProgress,
    Failed,
}

pub fn update_state(labels: MenuLabels, running: bool) {
    NATIVE_MENU.with(|menu| {
        if let Some(menu) = menu.borrow().as_ref() {
            unsafe {
                set_status_title(
                    &menu.service_item,
                    &labels.service_status,
                    labels.service_emphasis,
                );
                set_item_title(&menu.model_item, &labels.model_status);
                let (title, tag) = if running {
                    (&labels.stop_service, 3)
                } else {
                    (&labels.start_service, 2)
                };
                set_item_title(&menu.service_action_item, title);
                set_item_tag(&menu.service_action_item, tag);
                let service_action_item: id = *menu.service_action_item;
                service_action_item
                    .setEnabled_(labels.service_emphasis != StatusEmphasis::InProgress);
            }
        }
    });
}

pub fn set_window_hidden(hidden: bool, labels: &MenuLabels) {
    NATIVE_MENU.with(|menu| {
        if let Some(menu) = menu.borrow().as_ref() {
            unsafe {
                let (title, tag) = if hidden {
                    (&labels.show_window, 1)
                } else {
                    (&labels.hide_window, 6)
                };
                set_item_title(&menu.window_action_item, title);
                set_item_tag(&menu.window_action_item, tag);
            }
        }
    });
}

unsafe fn install_native_menu(labels: MenuLabels, running: bool) {
    unsafe {
        let status_bar = NSStatusBar::systemStatusBar(nil);
        let status_item =
            StrongPtr::retain(status_bar.statusItemWithLength_(NSVariableStatusItemLength));
        let menu = StrongPtr::retain(NSMenu::alloc(nil).initWithTitle_(ns_string("")));
        menu.setAutoenablesItems(NO);
        let target = StrongPtr::new(msg_send![target_class(), new]);

        let service_item = add_menu_item(&menu, &target, 0, &labels.service_status, true);
        set_status_title(
            &service_item,
            &labels.service_status,
            labels.service_emphasis,
        );
        let model_item = add_menu_item(&menu, &target, 0, &labels.model_status, true);
        menu.addItem_(NSMenuItem::separatorItem(nil));
        let window_action_item = add_menu_item(&menu, &target, 6, &labels.hide_window, true);
        let (service_action_title, service_action_tag) = if running {
            (&labels.stop_service, 3)
        } else {
            (&labels.start_service, 2)
        };
        let service_action_item = add_menu_item(
            &menu,
            &target,
            service_action_tag,
            service_action_title,
            labels.service_emphasis != StatusEmphasis::InProgress,
        );
        menu.addItem_(NSMenuItem::separatorItem(nil));
        add_menu_item(&menu, &target, 5, &labels.quit, true);

        status_item.setMenu_(*menu);
        let button = status_item.button();
        let image = NSImage::alloc(nil).initWithData_(ns_data(STATUS_ICON));
        if image != nil {
            // A template image lets macOS choose the correct foreground color
            // for the current light/dark menu bar appearance.
            let _: () = msg_send![image, setSize: cocoa::foundation::NSSize::new(20.0, 20.0)];
            let _: () = msg_send![image, setTemplate: YES];
            button.setImage_(image);
            let _: () = msg_send![button, setToolTip: ns_string(&labels.app_name)];
        }

        NATIVE_MENU.with(|slot| {
            *slot.borrow_mut() = Some(NativeMenu {
                _status_item: status_item,
                _menu: menu,
                _target: target,
                service_item,
                model_item,
                window_action_item,
                service_action_item,
            });
        });
    }
}

unsafe fn add_menu_item(
    menu: &StrongPtr,
    target: &StrongPtr,
    tag: isize,
    title: &str,
    enabled: bool,
) -> StrongPtr {
    unsafe {
        let item = NSMenuItem::alloc(nil).initWithTitle_action_keyEquivalent_(
            ns_string(title),
            sel!(handleMenuAction:),
            ns_string(""),
        );
        let target_id: id = **target;
        let _: () = msg_send![item, setTarget: target_id];
        let _: () = msg_send![item, setTag: tag];
        let _: () = msg_send![item, setEnabled: enabled as i8];
        menu.addItem_(item);
        StrongPtr::retain(item)
    }
}

unsafe fn set_item_title(item: &StrongPtr, title: &str) {
    let item: id = **item;
    item.setTitle_(ns_string(title));
}

unsafe fn set_item_tag(item: &StrongPtr, tag: isize) {
    let item: id = **item;
    let _: () = msg_send![item, setTag: tag];
}

unsafe fn set_status_title(item: &StrongPtr, title: &str, emphasis: StatusEmphasis) {
    let item: id = **item;
    let system_color: id = unsafe {
        match emphasis {
            StatusEmphasis::Neutral => msg_send![class!(NSColor), systemRedColor],
            StatusEmphasis::Running => msg_send![class!(NSColor), systemGreenColor],
            StatusEmphasis::InProgress => msg_send![class!(NSColor), systemOrangeColor],
            StatusEmphasis::Failed => msg_send![class!(NSColor), systemRedColor],
        }
    };
    let color = system_color;
    let text_color: id = unsafe { msg_send![class!(NSColor), labelColor] };
    let attributes: id = unsafe {
        msg_send![class!(NSDictionary), dictionaryWithObject: text_color forKey: ns_string("NSColor")]
    };
    let allocated: id = unsafe { msg_send![class!(NSMutableAttributedString), alloc] };
    let attributed_title: id = unsafe {
        msg_send![allocated, initWithString: ns_string(&format!("●  {title}")) attributes: attributes]
    };
    let _: () = unsafe {
        msg_send![attributed_title, addAttribute: ns_string("NSColor") value: color range: NSRange::new(0, 1)]
    };
    let _: () = unsafe { msg_send![item, setAttributedTitle: attributed_title] };
    let _: id = unsafe { msg_send![attributed_title, autorelease] };
}

fn ns_string(value: &str) -> id {
    unsafe { NSString::alloc(nil).init_str(value).autorelease() }
}

fn ns_data(bytes: &[u8]) -> id {
    unsafe { msg_send![class!(NSData), dataWithBytes: bytes.as_ptr() length: bytes.len()] }
}

fn target_class() -> *const Class {
    *TARGET_CLASS.get_or_init(|| unsafe {
        let mut declaration = ClassDecl::new("MiniMLXStatusMenuTarget", class!(NSObject))
            .expect("menubar target class should be unique");
        declaration.add_method(
            sel!(handleMenuAction:),
            handle_menu_action as extern "C" fn(&Object, Sel, id),
        );
        declaration.register() as *const Class as usize
    }) as *const Class
}

extern "C" fn handle_menu_action(_: &Object, _: Sel, item: id) {
    let tag: isize = unsafe { msg_send![item, tag] };
    let action = match tag {
        1 => MenuAction::ShowWindow,
        2 => MenuAction::StartService,
        3 => MenuAction::StopService,
        5 => MenuAction::Quit,
        6 => MenuAction::HideWindow,
        _ => return,
    };
    if let Some(sender) = MENU_SENDER.get() {
        let _ = sender.send(action);
    }
}
